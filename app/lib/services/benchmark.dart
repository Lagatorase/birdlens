import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'classifier.dart';
import 'preprocessing.dart';

/// Timing results from one benchmark run.
class BenchmarkResult {
  const BenchmarkResult({
    required this.modelBytes,
    required this.variant,
    required this.warmUp,
    required this.runs,
    required this.average,
    required this.p95,
    required this.fastest,
  });

  final int modelBytes;
  final String variant;

  /// The first inference, which includes one-off setup work.
  final Duration warmUp;
  final int runs;
  final Duration average;
  final Duration p95;
  final Duration fastest;
}

/// Average, p95 and fastest of a list of timings.
({Duration average, Duration p95, Duration fastest}) summarise(List<Duration> times) {
  if (times.isEmpty) throw ArgumentError('No timings to summarise.');
  final micros = times.map((t) => t.inMicroseconds).toList()..sort();
  final mean = micros.reduce((a, b) => a + b) / micros.length;
  // Nearest-rank p95: the smallest value with at least 95 percent of runs at
  // or below it.
  final rank = (0.95 * micros.length).ceil().clamp(1, micros.length);
  return (
    average: Duration(microseconds: mean.round()),
    p95: Duration(microseconds: micros[rank - 1]),
    fastest: Duration(microseconds: micros.first),
  );
}

/// Bundled photos the benchmark runs on.
const benchmarkSampleAssets = [
  'assets/samples/sample_1.jpg',
  'assets/samples/sample_2.jpg',
  'assets/samples/sample_3.jpg',
];

/// Time the model on the bundled samples. Preprocessing is done up front so
/// the numbers only cover the model itself.
Future<BenchmarkResult> runBenchmark(
  Classifier classifier, {
  int runs = 50,
  List<String> sampleAssets = benchmarkSampleAssets,
  AssetBundle? bundle,
  void Function(int done, int total)? onProgress,
}) async {
  final assets = bundle ?? rootBundle;
  final size = classifier.info.inputSize;
  final inputs = <Float32List>[];
  for (final asset in sampleAssets) {
    final data = await assets.load(asset);
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    inputs.add(await Isolate.run(() => preprocessBytes(bytes, size)));
  }
  // The first call pays for memory allocation and kernel setup, so I report
  // it on its own as the warm-up time.
  final warmUp = await classifier.timeInference(inputs.first);
  final times = <Duration>[];
  for (var i = 0; i < runs; i++) {
    // I cycle through the samples so it isn't the same image every time.
    times.add(await classifier.timeInference(inputs[i % inputs.length]));
    onProgress?.call(i + 1, runs);
  }
  final stats = summarise(times);
  return BenchmarkResult(
    modelBytes: classifier.modelBytes,
    variant: classifier.info.variant,
    warmUp: warmUp,
    runs: runs,
    average: stats.average,
    p95: stats.p95,
    fastest: stats.fastest,
  );
}
