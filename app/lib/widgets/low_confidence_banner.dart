import 'package:flutter/material.dart';

import '../utils/formatting.dart';

/// Shown when the top guess falls below the confidence threshold.
class LowConfidenceBanner extends StatelessWidget {
  const LowConfidenceBanner({super.key, required this.threshold});

  final double threshold;

  static const message =
      'Not confident, this may not be one of the 200 species I know.';

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      color: scheme.tertiaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.help_outline, color: scheme.onTertiaryContainer),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(message,
                      style: Theme.of(context)
                          .textTheme
                          .titleSmall
                          ?.copyWith(color: scheme.onTertiaryContainer)),
                  const SizedBox(height: 4),
                  Text(
                    'The top guess is under ${formatPercent(threshold)}. '
                    'Try a sharper, closer photo of a single bird.',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: scheme.onTertiaryContainer),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
