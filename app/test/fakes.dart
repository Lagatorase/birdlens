import 'dart:typed_data';

import 'package:birdlens/app_services.dart';
import 'package:birdlens/models/model_info.dart';
import 'package:birdlens/models/prediction.dart';
import 'package:birdlens/services/classifier.dart';
import 'package:birdlens/services/history_repository.dart';
import 'package:birdlens/services/image_store.dart';
import 'package:birdlens/services/photo_source.dart';
import 'package:birdlens/services/result_parser.dart';
import 'package:flutter/material.dart';

/// Model info that looks like the real export.
const fakeInfo = ModelInfo(
  variant: 'float16',
  modelAsset: 'assets/models/birdlens.tflite',
  labelsAsset: 'assets/labels.txt',
  inputSize: 224,
  threshold: 0.5,
  numClasses: 4,
  testTop1: 0.8,
  testTop5: 0.95,
);

const fakeLabels = ['Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Crested Auklet'];

/// A classifier that returns fixed probabilities without a model.
class FakeClassifier implements Classifier {
  FakeClassifier(this.probabilities, {this.error});

  final List<double> probabilities;
  final Object? error;

  @override
  ModelInfo get info => fakeInfo;
  @override
  List<String> get labels => fakeLabels;
  @override
  int get modelBytes => 12 * 1024 * 1024;

  @override
  Future<ClassificationResult> classifyBytes(Uint8List imageBytes) async {
    if (error != null) throw error!;
    return parseResult(probabilities, labels,
        threshold: info.threshold, inferenceTime: const Duration(milliseconds: 42));
  }

  @override
  Future<Duration> timeInference(Float32List input) async => const Duration(milliseconds: 10);

  @override
  Future<void> close() async {}
}

class FakePhotoSource implements PhotoSource {
  FakePhotoSource({this.bytes, this.error});
  final Uint8List? bytes;
  final Object? error;
  bool openedSettings = false;

  @override
  Future<Uint8List?> pick(PhotoOrigin origin) async {
    if (error != null) throw error!;
    return bytes;
  }

  @override
  Future<void> openSettings() async => openedSettings = true;
}

class FakeImageStore implements ImageStore {
  @override
  Future<StoredImage> save(Uint8List bytes) async =>
      const StoredImage('/missing/photo.jpg', '/missing/thumb.jpg');
}

/// A 1x1 transparent PNG, enough for Image.memory to decode.
final tinyPng = Uint8List.fromList(const [
  0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
  0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00,
  0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
  0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
  0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
]);

/// Wrap a widget with the app's services and theme for a widget test.
Widget withServices(
  Widget child, {
  Classifier? classifier,
  Object? loadError,
  HistoryRepository? history,
  PhotoSource? photos,
}) {
  final model = ClassifierController(() async {
    if (loadError != null) throw loadError;
    return classifier ?? FakeClassifier([0.7, 0.2, 0.05, 0.05]);
  });
  return AppServices(
    model: model..load(),
    history: history ?? InMemoryHistoryRepository(),
    images: FakeImageStore(),
    photos: photos ?? FakePhotoSource(),
    child: MaterialApp(home: child),
  );
}
