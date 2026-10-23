import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../app_services.dart';
import '../utils/formatting.dart';

/// Where the source code lives.
const repoUrl = 'https://github.com/Lagatorase/birdlens';

/// What the model is, how well it does, and where it falls short.
class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final info = AppServices.of(context).model.classifier?.info;
    final text = Theme.of(context).textTheme;
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const _Section(
            title: 'What this is',
            body: 'BirdLens identifies birds in photos using a neural network that '
                'runs on your phone. Photos never leave the device.',
          ),
          _Section(
            title: 'The model',
            body: 'An EfficientNetV2-B0 network, pretrained on ImageNet and fine-tuned on '
                'the CUB-200-2011 dataset of 200 bird species. '
                '${info == null ? '' : 'This build uses the ${info.variant} TensorFlow Lite variant. '}'
                '${info?.testTop1 == null ? '' : 'On the CUB test set it gets the species right '
                    '${formatPercent(info!.testTop1!)} of the time on the first guess'
                    '${info.testTop5 == null ? '' : ' and ${formatPercent(info.testTop5!)} within its top five'}.'}',
          ),
          const _Section(
            title: 'Limitations',
            body: '• It only knows 200 species, mostly North American. Any other bird gets '
                'matched to the closest of those 200.\n'
                '• Blurry, distant or partly hidden birds are often misidentified.\n'
                '• Some species look almost the same, and it mixes them up most often. '
                'On the test set, the worst were terns, gulls, flycatchers, vireos '
                'and jaegers.\n'
                '• The confidence score is the model\'s own estimate. It can still be '
                'confidently wrong.',
          ),
          const _Section(
            title: 'Data',
            body: 'Trained on CUB-200-2011 (Wah et al., Caltech). Its images come from Flickr '
                'and are for non-commercial research use. The dataset is not included in '
                'this app.',
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: () => launchUrl(Uri.parse(repoUrl), mode: LaunchMode.externalApplication),
            icon: const Icon(Icons.code),
            label: const Text('Source code on GitHub'),
          ),
          const SizedBox(height: 16),
          Text('Made by Lucas Lee, Temasek Polytechnic.',
              style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.body});
  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: text.titleMedium),
          const SizedBox(height: 6),
          Text(body, style: text.bodyMedium?.copyWith(height: 1.4)),
        ],
      ),
    );
  }
}
