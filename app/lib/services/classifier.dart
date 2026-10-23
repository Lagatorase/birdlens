import 'dart:convert';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/model_info.dart';
import '../models/prediction.dart';
import 'preprocessing.dart';
import 'result_parser.dart';

/// Thrown when the model or its labels can't be loaded.
class ModelLoadException implements Exception {
  const ModelLoadException(this.message);
  final String message;

  @override
  String toString() => 'ModelLoadException: $message';
}

/// What the screens need from a classifier. Widget tests use a fake.
abstract class Classifier {
  ModelInfo get info;
  List<String> get labels;

  /// Size of the .tflite file in bytes.
  int get modelBytes;

  /// Decode, preprocess and classify one photo.
  Future<ClassificationResult> classifyBytes(Uint8List imageBytes);

  /// Run the model on an already preprocessed image and return the time it
  /// took, for the benchmark screen.
  Future<Duration> timeInference(Float32List input);

  Future<void> close();
}

/// The real classifier, running the bundled TFLite model with tflite_flutter.
class TfliteClassifier implements Classifier {
  TfliteClassifier._(this.info, this.labels, this.modelBytes, this._interpreter,
      this._isolate);

  @override
  final ModelInfo info;
  @override
  final List<String> labels;
  @override
  final int modelBytes;
  final Interpreter _interpreter;
  final IsolateInterpreter _isolate;

  static const configAsset = 'assets/model_config.json';

  /// Load the config, labels and model from the app's assets.
  static Future<TfliteClassifier> load({AssetBundle? bundle}) async {
    final assets = bundle ?? rootBundle;
    try {
      // I read the config first because it names the model and label files.
      final config = ModelInfo.fromJson(
          jsonDecode(await assets.loadString(configAsset)) as Map<String, dynamic>);
      final labels = parseLabels(await assets.loadString(config.labelsAsset));
      if (labels.length != config.numClasses) {
        throw ModelLoadException(
            'Expected ${config.numClasses} labels but found ${labels.length}.');
      }
      final modelData = await assets.load(config.modelAsset);
      final buffer = modelData.buffer
          .asUint8List(modelData.offsetInBytes, modelData.lengthInBytes);
      // Four threads is a fair default for mid-range phones.
      final interpreter =
          Interpreter.fromBuffer(buffer, options: InterpreterOptions()..threads = 4);
      // I check the model's input matches the size in the config, so a wrong
      // export fails here instead of giving silent garbage later.
      final shape = interpreter.getInputTensor(0).shape;
      if (shape.length != 4 ||
          shape[1] != config.inputSize ||
          shape[2] != config.inputSize ||
          shape[3] != 3) {
        throw ModelLoadException('Unexpected model input shape $shape.');
      }
      // Inference runs in a background isolate so the UI never freezes.
      final isolate = await IsolateInterpreter.create(address: interpreter.address);
      return TfliteClassifier._(config, labels, buffer.length, interpreter, isolate);
    } on ModelLoadException {
      rethrow;
    } catch (e) {
      throw ModelLoadException('Could not load the bird model: $e');
    }
  }

  // IsolateInterpreter quietly skips a run if one is already in flight, so I
  // chain every call onto this future to run them one at a time.
  Future<void> _queue = Future.value();

  /// Run inference on HWC float pixels and return the class probabilities.
  Future<List<double>> runProbabilities(Float32List pixels) {
    final result = _queue.then((_) => _runOnce(pixels));
    // I keep the queue going even if this run throws.
    _queue = result.then((_) {}, onError: (_) {});
    return result;
  }

  Future<List<double>> _runOnce(Float32List pixels) async {
    final inputTensor = _interpreter.getInputTensor(0);
    final outputTensor = _interpreter.getOutputTensor(0);
    // I pass raw bytes in the type this model variant expects. tflite_flutter
    // copies a Uint8List straight into the tensor without reshaping.
    final Uint8List input;
    switch (inputTensor.type) {
      case TensorType.float32:
        input = pixels.buffer.asUint8List(pixels.offsetInBytes, pixels.lengthInBytes);
      case TensorType.uint8:
        final q = inputTensor.params;
        input = quantize(pixels, q.scale, q.zeroPoint, signed: false) as Uint8List;
      case TensorType.int8:
        final q = inputTensor.params;
        final signed = quantize(pixels, q.scale, q.zeroPoint, signed: true) as Int8List;
        input = signed.buffer.asUint8List();
      default:
        throw ModelLoadException('Unsupported input type ${inputTensor.type}.');
    }
    // The isolate copies the output into a ByteBuffer, which I then read in
    // the output tensor's own type.
    final output = ByteData(outputTensor.numBytes()).buffer;
    await _isolate.run(input, output);
    switch (outputTensor.type) {
      case TensorType.float32:
        return output.asFloat32List().toList();
      case TensorType.uint8:
        final q = outputTensor.params;
        return dequantize(output.asUint8List(), q.scale, q.zeroPoint);
      case TensorType.int8:
        final q = outputTensor.params;
        return dequantize(output.asInt8List(), q.scale, q.zeroPoint);
      default:
        throw ModelLoadException('Unsupported output type ${outputTensor.type}.');
    }
  }

  @override
  Future<ClassificationResult> classifyBytes(Uint8List imageBytes) async {
    final size = info.inputSize;
    // Decoding and resizing a photo takes a moment, so it runs off the UI thread.
    final pixels = await Isolate.run(() => preprocessBytes(imageBytes, size));
    final watch = Stopwatch()..start();
    final probabilities = await runProbabilities(pixels);
    watch.stop();
    return parseResult(probabilities, labels,
        threshold: info.threshold, inferenceTime: watch.elapsed);
  }

  @override
  Future<Duration> timeInference(Float32List input) async {
    final watch = Stopwatch()..start();
    await runProbabilities(input);
    watch.stop();
    return watch.elapsed;
  }

  @override
  Future<void> close() async {
    await _isolate.close();
    _interpreter.close();
  }
}
