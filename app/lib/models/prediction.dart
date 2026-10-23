/// One species guess from the model.
class Prediction {
  const Prediction({
    required this.index,
    required this.label,
    required this.confidence,
  });

  final int index;
  final String label;

  /// Softmax probability between 0 and 1.
  final double confidence;

  Map<String, Object> toJson() =>
      {'index': index, 'label': label, 'confidence': confidence};

  factory Prediction.fromJson(Map<String, dynamic> json) => Prediction(
        index: json['index'] as int,
        label: json['label'] as String,
        confidence: (json['confidence'] as num).toDouble(),
      );

  @override
  String toString() => 'Prediction($label, ${confidence.toStringAsFixed(3)})';
}

/// What the results screen needs from one classification.
class ClassificationResult {
  const ClassificationResult({
    required this.top,
    required this.threshold,
    this.inferenceTime,
  });

  /// Best guesses, highest confidence first.
  final List<Prediction> top;

  /// Below this top-1 confidence I show the "not confident" message.
  final double threshold;

  final Duration? inferenceTime;

  Prediction get best => top.first;

  bool get isConfident => best.confidence >= threshold;
}
