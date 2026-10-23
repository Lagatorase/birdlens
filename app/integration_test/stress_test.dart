// Runs on the emulator: many inferences in a row must all give a valid
// probability vector. I added this after an intermittent "noType" failure
// when the app used tflite_flutter's IsolateInterpreter.
//
// flutter test integration_test/stress_test.dart -d emulator-5554
import 'dart:convert';

import 'package:birdlens/services/classifier.dart';
import 'package:birdlens/services/preprocessing.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'fixture_data.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test('200 inferences in a row stay valid and identical', () async {
    final classifier = await TfliteClassifier.load();
    final bytes = base64Decode(fixtureImagesBase64['fixture_1.png']!);
    final pixels = preprocessBytes(bytes, classifier.info.inputSize);
    final first = await classifier.runProbabilities(pixels);
    final firstTop = _argmax(first);
    for (var i = 0; i < 200; i++) {
      final probs = await classifier.runProbabilities(pixels);
      expect(probs.length, 200);
      // A softmax output should sum to 1 and never be negative.
      expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-3), reason: 'run $i');
      expect(probs.every((p) => p >= 0), isTrue, reason: 'run $i');
      expect(_argmax(probs), firstTop, reason: 'run $i');
    }
    await classifier.close();
  });

  test('two classifiers can load and close one after another', () async {
    // The app reloads the model on "Try again", so loading twice must work.
    for (var i = 0; i < 2; i++) {
      final classifier = await TfliteClassifier.load();
      expect(classifier.labels.length, 200);
      await classifier.close();
    }
  });
}

int _argmax(List<double> values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
