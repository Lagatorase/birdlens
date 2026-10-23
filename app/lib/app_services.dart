import 'package:flutter/widgets.dart';

import 'services/classifier.dart';
import 'services/history_repository.dart';
import 'services/image_store.dart';
import 'services/photo_source.dart';

enum ModelStatus { loading, ready, failed }

/// Loads the classifier once and lets screens react to its state.
class ClassifierController extends ChangeNotifier {
  ClassifierController(this._loader);

  final Future<Classifier> Function() _loader;

  ModelStatus status = ModelStatus.loading;
  Classifier? classifier;
  String? error;

  Future<void> load() async {
    status = ModelStatus.loading;
    error = null;
    notifyListeners();
    try {
      classifier = await _loader();
      status = ModelStatus.ready;
    } catch (e) {
      // I keep the message so the home screen can show it with a retry button.
      error = e.toString();
      status = ModelStatus.failed;
    }
    notifyListeners();
  }

  @override
  void dispose() {
    classifier?.close();
    super.dispose();
  }
}

/// Hands the app's services down the widget tree without a DI package.
class AppServices extends InheritedWidget {
  const AppServices({
    super.key,
    required this.model,
    required this.history,
    required this.images,
    required this.photos,
    required super.child,
  });

  final ClassifierController model;
  final HistoryRepository history;
  final ImageStore images;
  final PhotoSource photos;

  static AppServices of(BuildContext context) {
    final services = context.dependOnInheritedWidgetOfExactType<AppServices>();
    assert(services != null, 'AppServices is missing above this widget');
    return services!;
  }

  @override
  bool updateShouldNotify(AppServices oldWidget) =>
      model != oldWidget.model ||
      history != oldWidget.history ||
      images != oldWidget.images ||
      photos != oldWidget.photos;
}
