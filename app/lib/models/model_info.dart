/// Facts about the bundled model, read from assets/model_config.json.
///
/// The Python export script writes that file, so the app and the README
/// always describe the same model.
class ModelInfo {
  const ModelInfo({
    required this.variant,
    required this.modelAsset,
    required this.labelsAsset,
    required this.inputSize,
    required this.threshold,
    required this.numClasses,
    this.testTop1,
    this.testTop5,
  });

  final String variant;
  final String modelAsset;
  final String labelsAsset;
  final int inputSize;

  /// Top-1 confidence below this counts as "not confident".
  final double threshold;
  final int numClasses;

  /// Accuracy on the CUB test set, measured with the .tflite file in Python.
  final double? testTop1;
  final double? testTop5;

  factory ModelInfo.fromJson(Map<String, dynamic> json) => ModelInfo(
        variant: json['variant'] as String,
        modelAsset: json['model_asset'] as String,
        labelsAsset: json['labels_asset'] as String,
        inputSize: json['input_size'] as int,
        threshold: (json['threshold'] as num).toDouble(),
        numClasses: json['num_classes'] as int,
        testTop1: (json['test_top1'] as num?)?.toDouble(),
        testTop5: (json['test_top5'] as num?)?.toDouble(),
      );
}
