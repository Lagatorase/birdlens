import 'dart:convert';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/model_info.dart';
import '../models/prediction.dart';
import 'inference_worker.dart';
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

  /// Run the model on an already preprocessed image and return how long the
  /// interpreter's invoke call took, for the benchmark screen.
  Future<Duration> timeInference(Float32List input);

  Future<void> close();
}

/// The real classifier: the bundled TFLite model running in a background
/// isolate through tflite_flutter.
class TfliteClassifier implements Classifier {
  TfliteClassifier._(this.info, this.labels, this.modelBytes, this._worker);

  @override
  final ModelInfo info;
  @override
  final List<String> labels;
  @override
  final int modelBytes;
  final InferenceWorker _worker;

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
      final worker = await InferenceWorker.start(buffer, threads: 4);
      // I check the model's input matches the config, so a wrong export fails
      // here instead of giving silent garbage later.
      final shape = worker.input.shape;
      if (shape.length != 4 ||
          shape[1] != config.inputSize ||
          shape[2] != config.inputSize ||
          shape[3] != 3) {
        await worker.close();
        throw ModelLoadException('Unexpected model input shape $shape.');
      }
      if (worker.output.shape.last != labels.length) {
        await worker.close();
        throw ModelLoadException(
            'Model has ${worker.output.shape.last} outputs but there are ${labels.length} labels.');
      }
      return TfliteClassifier._(config, labels, buffer.length, worker);
    } on ModelLoadException {
      rethrow;
    } catch (e) {
      throw ModelLoadException('Could not load the bird model: $e');
    }
  }

  /// Run the model on HWC float pixels and return the class probabilities.
  Future<List<double>> runProbabilities(Float32List pixels) async =>
      (await _run(pixels)).probabilities;

  Future<({List<double> probabilities, Duration invokeTime})> _run(Float32List pixels) async {
    final input = _worker.input;
    // I convert the pixels to the bytes this model variant expects.
    final Uint8List bytes;
    switch (input.type) {
      case TensorType.float32:
        bytes = pixels.buffer.asUint8List(pixels.offsetInBytes, pixels.lengthInBytes);
      case TensorType.uint8:
        bytes = quantize(pixels, input.scale, input.zeroPoint, signed: false) as Uint8List;
      case TensorType.int8:
        final signed = quantize(pixels, input.scale, input.zeroPoint, signed: true) as Int8List;
        bytes = signed.buffer.asUint8List();
      default:
        throw ModelLoadException('Unsupported input type ${input.type}.');
    }
    final result = await _worker.run(bytes);
    // Then I read the output bytes back in the output tensor's own type.
    final out = _worker.output;
    final raw = result.bytes;
    final List<double> probabilities;
    switch (out.type) {
      case TensorType.float32:
        // I copy into an aligned buffer because the bytes may not start on a
        // 4-byte boundary.
        probabilities = Float32List.fromList(
                Uint8List.fromList(raw).buffer.asFloat32List())
            .toList();
      case TensorType.uint8:
        probabilities = dequantize(raw, out.scale, out.zeroPoint);
      case TensorType.int8:
        probabilities = dequantize(Int8List.fromList(raw), out.scale, out.zeroPoint);
      default:
        throw ModelLoadException('Unsupported output type ${out.type}.');
    }
    return (probabilities: probabilities, invokeTime: result.invokeTime);
  }

  @override
  Future<ClassificationResult> classifyBytes(Uint8List imageBytes) async {
    final size = info.inputSize;
    // Decoding and resizing a photo takes a moment, so it runs off the UI thread.
    final pixels = await Isolate.run(() => preprocessBytes(imageBytes, size));
    final result = await _run(pixels);
    return parseResult(result.probabilities, labels,
        threshold: info.threshold, inferenceTime: result.invokeTime);
  }

  @override
  Future<Duration> timeInference(Float32List input) async => (await _run(input)).invokeTime;

  @override
  Future<void> close() => _worker.close();
}
