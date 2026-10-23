import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

/// Tensor facts read once, inside the worker, right after allocation.
class TensorInfo {
  const TensorInfo(this.type, this.shape, this.scale, this.zeroPoint, this.numBytes);

  final TensorType type;
  final List<int> shape;
  final double scale;
  final int zeroPoint;
  final int numBytes;
}

/// What one inference sends back: raw output bytes and the native invoke time.
class InferenceOutput {
  const InferenceOutput(this.bytes, this.invokeTime);
  final Uint8List bytes;
  final Duration invokeTime;
}

/// Runs the TFLite interpreter in one long-lived background isolate.
///
/// I first used tflite_flutter's IsolateInterpreter, but on the emulator a
/// benchmark run sometimes failed with an output tensor type of "noType".
/// That class shares one native interpreter between two isolates and
/// re-allocates its tensors on every call, and my code read tensor details
/// on the main isolate through handles that could go stale. Here only the
/// worker ever touches the interpreter: it allocates once, reports the
/// tensor details once, and then just swaps bytes in and out.
class InferenceWorker {
  InferenceWorker._(this._isolate, this._requests, this._responses, this.input, this.output);

  final Isolate _isolate;
  final SendPort _requests;
  final StreamIterator<dynamic> _responses;
  final TensorInfo input;
  final TensorInfo output;

  // One request at a time, so replies always match their request.
  Future<void> _queue = Future.value();

  static Future<InferenceWorker> start(Uint8List modelBytes, {int threads = 4}) async {
    final fromWorker = ReceivePort();
    final isolate = await Isolate.spawn(
      _workerMain,
      [fromWorker.sendPort, TransferableTypedData.fromList([modelBytes]), threads],
      debugName: 'BirdLensInference',
    );
    final responses = StreamIterator<dynamic>(fromWorker);
    // The worker's first message is its SendPort, the second is either the
    // tensor details or an error string from loading the model.
    await responses.moveNext();
    final requests = responses.current as SendPort;
    await responses.moveNext();
    final ready = responses.current;
    if (ready is String) {
      isolate.kill();
      fromWorker.close();
      throw StateError(ready);
    }
    final infos = ready as List;
    return InferenceWorker._(
        isolate, requests, responses, infos[0] as TensorInfo, infos[1] as TensorInfo);
  }

  /// Run one inference. [inputBytes] must already be in the input tensor's type.
  Future<InferenceOutput> run(Uint8List inputBytes) {
    final result = _queue.then((_) async {
      _requests.send(TransferableTypedData.fromList([inputBytes]));
      await _responses.moveNext();
      final reply = _responses.current;
      if (reply is String) throw StateError(reply);
      final parts = reply as List;
      return InferenceOutput(
        (parts[0] as TransferableTypedData).materialize().asUint8List(),
        Duration(microseconds: parts[1] as int),
      );
    });
    // I keep the queue alive even when one run fails.
    _queue = result.then((_) {}, onError: (_) {});
    return result;
  }

  Future<void> close() async {
    _requests.send(null);
    await _responses.cancel();
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }
}

TensorInfo _describe(Tensor t) =>
    TensorInfo(t.type, t.shape, t.params.scale, t.params.zeroPoint, t.numBytes());

/// The worker isolate's entry point.
Future<void> _workerMain(List<Object> args) async {
  final toMain = args[0] as SendPort;
  final modelBytes = (args[1] as TransferableTypedData).materialize().asUint8List();
  final threads = args[2] as int;
  final inbox = ReceivePort();
  toMain.send(inbox.sendPort);

  final Interpreter interpreter;
  try {
    interpreter = Interpreter.fromBuffer(modelBytes,
        options: InterpreterOptions()..threads = threads);
    // I allocate once here.
    interpreter.allocateTensors();
    toMain.send([_describe(interpreter.getInputTensor(0)), _describe(interpreter.getOutputTensor(0))]);
  } catch (e) {
    toMain.send('Could not load the model: $e');
    return;
  }

  // I never keep Tensor objects between calls. On the emulator, handles I
  // fetched before the first invoke went stale (the runtime seems to rebuild
  // its tensors when it applies the default XNNPACK delegate), which showed
  // up as a 0-byte input and a crash on a null output buffer. Asking the
  // interpreter for fresh handles each time avoids that.
  await for (final message in inbox) {
    // null is my signal to shut down.
    if (message == null) break;
    // I report which step failed, with sizes, so a failure is easy to read.
    var step = 'copying the input';
    try {
      final bytes = (message as TransferableTypedData).materialize().asUint8List();
      final inputTensor = interpreter.getInputTensor(0);
      if (bytes.length != inputTensor.numBytes()) {
        throw StateError('input is ${bytes.length} bytes, tensor wants ${inputTensor.numBytes()}');
      }
      inputTensor.setTo(bytes);
      step = 'running the model';
      final watch = Stopwatch()..start();
      interpreter.invoke();
      watch.stop();
      step = 'reading the output';
      // copyTo goes through TfLiteTensorCopyToBuffer and returns a Dart copy.
      final outputTensor = interpreter.getOutputTensor(0);
      final out = outputTensor.copyTo(Uint8List(outputTensor.numBytes())) as Uint8List;
      toMain.send([TransferableTypedData.fromList([out]), watch.elapsedMicroseconds]);
    } catch (e) {
      toMain.send('Inference failed while $step: $e');
    }
  }
  interpreter.close();
  inbox.close();
}
