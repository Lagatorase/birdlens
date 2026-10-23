import 'package:flutter/material.dart';

import '../app_services.dart';
import '../services/benchmark.dart';
import '../services/classifier.dart';
import '../utils/formatting.dart';
import '../widgets/message_view.dart';

/// Times the model on bundled sample photos.
class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key, this.runner = runBenchmark});

  /// Swappable so widget tests don't need a real model.
  final Future<BenchmarkResult> Function(
    Classifier classifier, {
    void Function(int done, int total)? onProgress,
  }) runner;

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  BenchmarkResult? _result;
  Object? _error;
  bool _running = false;
  double _progress = 0;

  Future<void> _run() async {
    final classifier = AppServices.of(context).model.classifier;
    if (classifier == null) return;
    setState(() {
      _running = true;
      _error = null;
      _progress = 0;
    });
    try {
      final result = await widget.runner(classifier, onProgress: (done, total) {
        if (mounted) setState(() => _progress = done / total);
      });
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = e);
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final services = AppServices.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Benchmark')),
      body: ListenableBuilder(
        listenable: services.model,
        builder: (context, _) {
          final classifier = services.model.classifier;
          if (classifier == null) {
            return const MessageView(
              icon: Icons.hourglass_empty,
              title: 'Model not ready',
              body: 'The benchmark needs the model to load first.',
            );
          }
          final text = Theme.of(context).textTheme;
          final scheme = Theme.of(context).colorScheme;
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Model', style: text.titleSmall),
                      const SizedBox(height: 8),
                      _Row('Variant', classifier.info.variant),
                      _Row('File size', formatBytes(classifier.modelBytes)),
                      _Row('Input', '${classifier.info.inputSize} x ${classifier.info.inputSize} RGB'),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: _running ? null : _run,
                icon: const Icon(Icons.speed),
                label: Text(_running ? 'Running...' : 'Run benchmark'),
                style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              ),
              if (_running) ...[
                const SizedBox(height: 12),
                LinearProgressIndicator(value: _progress),
              ],
              if (_error != null) ...[
                const SizedBox(height: 12),
                Text('Benchmark failed: $_error', style: TextStyle(color: scheme.error)),
              ],
              if (_result != null) ...[
                const SizedBox(height: 12),
                _ResultCard(result: _result!),
              ],
              const SizedBox(height: 12),
              Text(
                'These timings depend on the device. They cover the model only, '
                'not decoding or resizing the photo, and will differ on other phones.',
                style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});
  final BenchmarkResult result;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Results on this device', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            _Row('Warm-up (first run)', formatMs(result.warmUp)),
            _Row('Average', formatMs(result.average)),
            _Row('p95', formatMs(result.p95)),
            _Row('Fastest', formatMs(result.fastest)),
            _Row('Timed runs', '${result.runs}'),
          ],
        ),
      ),
    );
  }
}

class _Row extends StatelessWidget {
  const _Row(this.label, this.value);
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(child: Text(label, style: text.bodyMedium)),
          Text(value,
              style: text.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                fontFeatures: const [FontFeature.tabularFigures()],
              )),
        ],
      ),
    );
  }
}
