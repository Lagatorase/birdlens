import 'package:flutter/material.dart';

import 'app.dart';
import 'app_services.dart';
import 'services/classifier.dart';
import 'services/history_repository.dart';
import 'services/image_store.dart';
import 'services/photo_source.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // I open the history database before the first frame so the History tab
  // never has to wait on it.
  final HistoryRepository history;
  try {
    history = await SqfliteHistoryRepository.open();
  } catch (_) {
    // If the database can't open, the app still works without saving scans.
    runApp(_withServices(InMemoryHistoryRepository()));
    return;
  }
  runApp(_withServices(history));
}

Widget _withServices(HistoryRepository history) {
  // The model loads in the background while the home screen shows a spinner.
  final model = ClassifierController(TfliteClassifier.load)..load();
  return AppServices(
    model: model,
    history: history,
    images: FileImageStore(),
    photos: ImagePickerPhotoSource(),
    child: const BirdLensApp(),
  );
}
