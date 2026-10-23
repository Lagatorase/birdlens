import 'package:flutter/material.dart';

import '../models/prediction.dart';
import '../utils/formatting.dart';

/// A species name with a horizontal bar showing how sure the model is.
class ConfidenceBar extends StatelessWidget {
  const ConfidenceBar({super.key, required this.prediction, required this.rank});

  final Prediction prediction;

  /// 1 for the top guess. I highlight it more strongly than the others.
  final int rank;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    final isTop = rank == 1;
    return Semantics(
      label: '${prediction.label}, ${formatPercent(prediction.confidence)} confidence',
      excludeSemantics: true,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    prediction.label,
                    style: isTop
                        ? text.titleMedium?.copyWith(fontWeight: FontWeight.w600)
                        : text.bodyLarge,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  formatPercent(prediction.confidence),
                  style: text.labelLarge?.copyWith(
                    color: isTop ? scheme.primary : scheme.onSurfaceVariant,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: prediction.confidence,
                minHeight: isTop ? 10 : 6,
                backgroundColor: scheme.surfaceContainerHighest,
                color: isTop ? scheme.primary : scheme.secondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
