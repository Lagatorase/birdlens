// Runs on the emulator: Dart preprocessing plus the real tflite_flutter
// runtime on the bundled model, compared with the Python LiteRT outputs.
//
// flutter test integration_test/parity_test.dart -d emulator-5554
import 'dart:convert';

import 'package:birdlens/services/classifier.dart';
import 'package:birdlens/services/preprocessing.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'fixture_data.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final doc = jsonDecode(fixturesJson) as Map<String, dynamic>;
  final shipped = doc['shipped_variant'] as String;
  late TfliteClassifier classifier;

  setUpAll(() async {
    classifier = await TfliteClassifier.load();
  });

  tearDownAll(() => classifier.close());

  test('the bundled model is the variant the fixtures expect', () {
    expect(classifier.info.variant, shipped);
  });

  for (final fixture in doc['fixtures'] as List) {
    final name = fixture['name'] as String;
    final inputs = fixture['inputs'] as Map<String, dynamic>;

    test('$name.png: on-device probabilities match Python', () async {
      final bytes = base64Decode(fixtureImagesBase64['$name.png']!);
      final probs = await classifier.runProbabilities(
          preprocessBytes(bytes, classifier.info.inputSize));
      final expected = (inputs['png']['outputs'][shipped]['probs'] as List).cast<num>();
      var maxDiff = 0.0;
      for (var i = 0; i < probs.length; i++) {
        final d = (probs[i] - expected[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      // Same input pixels and the same model file, so only float rounding
      // and kernel differences between the two runtimes remain.
      // I print the real gap so PROGRESS.md and the README can quote it.
      debugPrint('PARITY $name.png max_prob_diff=${maxDiff.toStringAsExponential(2)}');
      expect(maxDiff, lessThan(0.01), reason: 'largest probability difference');
      final top = argmax(probs);
      expect(top, (inputs['png']['outputs'][shipped]['top5'] as List).first);
    });

    test('$name.jpg: on-device top-1 matches Python', () async {
      final bytes = base64Decode(fixtureImagesBase64['$name.jpg']!);
      final probs = await classifier.runProbabilities(
          preprocessBytes(bytes, classifier.info.inputSize));
      final expected = inputs['jpg']['outputs'][shipped] as Map<String, dynamic>;
      // JPEG decoding differs slightly between Dart and Python, so I only
      // require the same top guess and a close top-1 probability.
      final wantTop1Index = (expected['top5'] as List).first as int;
      debugPrint('PARITY $name.jpg top1_match=${argmax(probs) == wantTop1Index} '
          'top1_prob_diff=${(probs[wantTop1Index] - ((expected['top5_probs'] as List).first as num)).abs().toStringAsFixed(4)}');
      expect(argmax(probs), wantTop1Index);
      final wantTop1 = ((expected['top5_probs'] as List).first as num).toDouble();
      expect(probs[argmax(probs)], closeTo(wantTop1, 0.05));
    });
  }
}

int argmax(List<double> values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
