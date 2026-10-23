// Checks that my Dart preprocessing matches the Python preprocessing on the
// same fixture images. The full model output check runs on the emulator in
// integration_test/parity_test.dart, because tflite_flutter has no Windows
// library for host tests.
import 'dart:convert';
import 'dart:io';

import 'package:birdlens/services/preprocessing.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final fixtureFile = File('test/fixtures/fixtures.json');
  if (!fixtureFile.existsSync()) {
    test('parity fixtures', () {}, skip: 'Run training/prepare_app_assets.py first');
    return;
  }
  final doc = jsonDecode(fixtureFile.readAsStringSync()) as Map<String, dynamic>;
  final size = doc['input_size'] as int;
  final probes = [
    for (final p in doc['probe_points_yx'] as List) [(p as List)[0] as int, p[1] as int],
  ];

  for (final fixture in doc['fixtures'] as List) {
    final name = fixture['name'] as String;
    final inputs = fixture['inputs'] as Map<String, dynamic>;

    test('$name.png: Dart preprocessing matches Python exactly', () {
      // PNG is lossless, so both sides start from identical pixels and only
      // the resize can differ. TensorFlow computes sample positions in float32
      // and Dart in float64, which moves values by about 0.01 at most.
      final pixels = preprocessBytes(
          File('test/fixtures/images/$name.png').readAsBytesSync(), size);
      final expected = inputs['png'] as Map<String, dynamic>;
      final probeValues = expected['probe_values'] as List;
      for (var i = 0; i < probes.length; i++) {
        final y = probes[i][0], x = probes[i][1];
        for (var c = 0; c < 3; c++) {
          final want = ((probeValues[i] as List)[c] as num).toDouble();
          expect(pixels[(y * size + x) * 3 + c], closeTo(want, 0.02),
              reason: 'pixel ($y, $x) channel $c');
        }
      }
      final means = channelMeans(pixels);
      final wantMeans = (expected['channel_means'] as List).cast<num>();
      for (var c = 0; c < 3; c++) {
        expect(means[c], closeTo(wantMeans[c].toDouble(), 0.01));
      }
    });

    test('$name.jpg: Dart preprocessing is close to Python', () {
      // JPEG decoders differ slightly (Python uses libjpeg-turbo, Dart's image
      // package has its own decoder), so I allow small per-pixel differences
      // but the image as a whole must match closely.
      final pixels = preprocessBytes(
          File('test/fixtures/images/$name.jpg').readAsBytesSync(), size);
      final expected = inputs['jpg'] as Map<String, dynamic>;
      final probeValues = expected['probe_values'] as List;
      var totalDiff = 0.0;
      for (var i = 0; i < probes.length; i++) {
        final y = probes[i][0], x = probes[i][1];
        for (var c = 0; c < 3; c++) {
          final want = ((probeValues[i] as List)[c] as num).toDouble();
          final diff = (pixels[(y * size + x) * 3 + c] - want).abs();
          expect(diff, lessThan(12), reason: 'pixel ($y, $x) channel $c');
          totalDiff += diff;
        }
      }
      expect(totalDiff / (probes.length * 3), lessThan(2.0));
      final means = channelMeans(pixels);
      final wantMeans = (expected['channel_means'] as List).cast<num>();
      for (var c = 0; c < 3; c++) {
        expect(means[c], closeTo(wantMeans[c].toDouble(), 1.0));
      }
    });
  }
}

List<double> channelMeans(List<double> pixels) {
  final sums = [0.0, 0.0, 0.0];
  for (var i = 0; i < pixels.length; i++) {
    sums[i % 3] += pixels[i];
  }
  return [for (final s in sums) s / (pixels.length / 3)];
}
