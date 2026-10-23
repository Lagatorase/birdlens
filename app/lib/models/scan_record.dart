import 'dart:convert';

import 'prediction.dart';

/// One saved scan in the history list.
class ScanRecord {
  const ScanRecord({
    this.id,
    required this.imagePath,
    required this.thumbnailPath,
    required this.predictions,
    required this.threshold,
    required this.createdAt,
  });

  final int? id;
  final String imagePath;
  final String thumbnailPath;

  /// The top-3 guesses at the time of the scan.
  final List<Prediction> predictions;
  final double threshold;
  final DateTime createdAt;

  Prediction get best => predictions.first;

  bool get isConfident => best.confidence >= threshold;

  ScanRecord copyWith({int? id}) => ScanRecord(
        id: id ?? this.id,
        imagePath: imagePath,
        thumbnailPath: thumbnailPath,
        predictions: predictions,
        threshold: threshold,
        createdAt: createdAt,
      );

  /// Row format for the sqflite table. I keep the top label and confidence
  /// in their own columns so they're easy to query later.
  Map<String, Object?> toRow() => {
        if (id != null) 'id': id,
        'image_path': imagePath,
        'thumbnail_path': thumbnailPath,
        'top_label': best.label,
        'top_confidence': best.confidence,
        'predictions_json': jsonEncode([for (final p in predictions) p.toJson()]),
        'threshold': threshold,
        'created_at': createdAt.millisecondsSinceEpoch,
      };

  factory ScanRecord.fromRow(Map<String, Object?> row) => ScanRecord(
        id: row['id'] as int?,
        imagePath: row['image_path'] as String,
        thumbnailPath: row['thumbnail_path'] as String,
        predictions: [
          for (final p in jsonDecode(row['predictions_json'] as String) as List)
            Prediction.fromJson(p as Map<String, dynamic>),
        ],
        threshold: (row['threshold'] as num).toDouble(),
        createdAt: DateTime.fromMillisecondsSinceEpoch(row['created_at'] as int),
      );
}
