import '../models/prediction.dart';

/// Turn integer model outputs back into probabilities.
List<double> dequantize(List<int> raw, double scale, int zeroPoint) {
  // real = scale * (q - zeroPoint), the standard TFLite affine mapping.
  return [for (final q in raw) scale * (q - zeroPoint)];
}

/// Pick the k most likely classes, highest first.
List<Prediction> topK(List<double> probabilities, List<String> labels, int k) {
  if (probabilities.length != labels.length) {
    throw ArgumentError(
        'Model gave ${probabilities.length} scores but there are ${labels.length} labels.');
  }
  // I sort indices instead of copying the scores so I keep the class ids.
  final indices = List<int>.generate(probabilities.length, (i) => i)
    ..sort((a, b) {
      final byScore = probabilities[b].compareTo(probabilities[a]);
      // Ties go to the lower class index so the order is stable.
      return byScore != 0 ? byScore : a.compareTo(b);
    });
  return [
    for (final i in indices.take(k))
      Prediction(
        index: i,
        label: labels[i],
        // Rounding or quantisation can push a score a hair outside 0 to 1.
        confidence: probabilities[i].clamp(0.0, 1.0),
      ),
  ];
}

/// Build the result the UI shows from raw probabilities.
ClassificationResult parseResult(
  List<double> probabilities,
  List<String> labels, {
  required double threshold,
  int k = 3,
  Duration? inferenceTime,
}) {
  return ClassificationResult(
    top: topK(probabilities, labels, k),
    threshold: threshold,
    inferenceTime: inferenceTime,
  );
}

/// Read labels.txt into a list, skipping blank lines.
List<String> parseLabels(String text) => [
      for (final line in text.split('\n'))
        if (line.trim().isNotEmpty) line.trim(),
    ];
