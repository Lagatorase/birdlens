import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../app_services.dart';
import '../models/prediction.dart';
import '../models/scan_record.dart';
import '../services/preprocessing.dart';
import '../utils/formatting.dart';
import '../widgets/confidence_bar.dart';
import '../widgets/low_confidence_banner.dart';
import '../widgets/message_view.dart';

/// Shows a photo with its top-3 guesses.
///
/// With [imageBytes] it classifies a new photo and saves it to history.
/// With [record] it shows a saved scan without running the model again.
class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, this.imageBytes, this.record})
      : assert((imageBytes == null) != (record == null));

  final Uint8List? imageBytes;
  final ScanRecord? record;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  Future<_Outcome>? _outcome;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // I start the work once, the first time the services are available.
    _outcome ??= _run();
  }

  Future<_Outcome> _run() async {
    final saved = widget.record;
    if (saved != null) {
      return _Outcome(saved.predictions, saved.threshold, null, saved: true);
    }
    final services = AppServices.of(context);
    final classifier = services.model.classifier!;
    final bytes = widget.imageBytes!;
    final result = await classifier.classifyBytes(bytes);
    // Saving is a bonus. If it fails I still show the result.
    var savedOk = true;
    try {
      final stored = await services.images.save(bytes);
      await services.history.add(ScanRecord(
        imagePath: stored.imagePath,
        thumbnailPath: stored.thumbnailPath,
        predictions: result.top,
        threshold: result.threshold,
        createdAt: DateTime.now(),
      ));
    } catch (_) {
      savedOk = false;
    }
    return _Outcome(result.top, result.threshold, result.inferenceTime, saved: savedOk);
  }

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    return Scaffold(
      appBar: AppBar(
        title: Text(record == null ? 'Result' : formatTimestamp(record.createdAt)),
      ),
      body: FutureBuilder<_Outcome>(
        future: _outcome,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            final error = snapshot.error;
            // A corrupt or unsupported file gets a friendly message.
            return MessageView(
              icon: Icons.broken_image_outlined,
              isError: true,
              title: error is ImageDecodeException
                  ? 'I could not read this image'
                  : 'Something went wrong',
              body: error is ImageDecodeException
                  ? 'The file may be damaged or in a format I do not support. '
                      'Try a different JPEG or PNG photo.'
                  : '$error',
              actionLabel: 'Back',
              onAction: () => Navigator.pop(context),
            );
          }
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _Photo(bytes: widget.imageBytes, path: record?.imagePath),
              const SizedBox(height: 16),
              if (!snapshot.hasData)
                const _Working()
              else
                ..._results(context, snapshot.data!),
            ],
          );
        },
      ),
    );
  }

  List<Widget> _results(BuildContext context, _Outcome outcome) {
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    final confident = outcome.top.first.confidence >= outcome.threshold;
    return [
      if (!confident) ...[
        LowConfidenceBanner(threshold: outcome.threshold),
        const SizedBox(height: 12),
      ],
      Card(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(confident ? 'Top matches' : 'Closest matches', style: text.titleSmall),
              const SizedBox(height: 4),
              for (var i = 0; i < outcome.top.length; i++)
                ConfidenceBar(prediction: outcome.top[i], rank: i + 1),
            ],
          ),
        ),
      ),
      const SizedBox(height: 12),
      // A short line of facts about this run.
      Wrap(
        spacing: 16,
        runSpacing: 4,
        children: [
          if (outcome.inferenceTime != null)
            _Fact(Icons.timer_outlined, 'Model time ${formatMs(outcome.inferenceTime!)}'),
          if (widget.record == null)
            _Fact(outcome.saved ? Icons.history : Icons.warning_amber,
                outcome.saved ? 'Saved to history' : 'Could not save to history'),
        ],
      ),
      const SizedBox(height: 8),
      Text(
        'Confidence is the model\'s own score, not a guarantee.',
        style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
      ),
    ];
  }
}

class _Outcome {
  const _Outcome(this.top, this.threshold, this.inferenceTime, {required this.saved});
  final List<Prediction> top;
  final double threshold;
  final Duration? inferenceTime;
  final bool saved;
}

class _Photo extends StatelessWidget {
  const _Photo({this.bytes, this.path});

  final Uint8List? bytes;
  final String? path;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // A placeholder covers files that can't be decoded or were deleted.
    Widget broken(BuildContext c, Object e, StackTrace? s) => Container(
          height: 200,
          color: scheme.surfaceContainerHighest,
          child: Icon(Icons.broken_image_outlined, size: 48, color: scheme.onSurfaceVariant),
        );
    final image = bytes != null
        ? Image.memory(bytes!, fit: BoxFit.contain, errorBuilder: broken)
        : Image.file(File(path!), fit: BoxFit.contain, errorBuilder: broken);
    // I round the photo itself rather than boxing it, so wide shots don't
    // get grey bars above and below.
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: 360),
      // heightFactor 1 makes the box only as tall as the photo.
      child: Align(
        heightFactor: 1,
        child: ClipRRect(borderRadius: BorderRadius.circular(16), child: image),
      ),
    );
  }
}

class _Working extends StatelessWidget {
  const _Working();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(32),
      child: Column(
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 16),
          Text('Looking at the bird...'),
        ],
      ),
    );
  }
}

class _Fact extends StatelessWidget {
  const _Fact(this.icon, this.label);
  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.onSurfaceVariant;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 4),
        Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: color)),
      ],
    );
  }
}
