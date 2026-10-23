import 'package:birdlens/models/prediction.dart';
import 'package:birdlens/models/scan_record.dart';
import 'package:birdlens/screens/about_screen.dart';
import 'package:birdlens/screens/benchmark_screen.dart';
import 'package:birdlens/screens/history_screen.dart';
import 'package:birdlens/screens/home_shell.dart';
import 'package:birdlens/screens/identify_screen.dart';
import 'package:birdlens/screens/result_screen.dart';
import 'package:birdlens/services/benchmark.dart';
import 'package:birdlens/services/history_repository.dart';
import 'package:birdlens/services/photo_source.dart';
import 'package:birdlens/services/preprocessing.dart';
import 'package:birdlens/widgets/low_confidence_banner.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'fakes.dart';

void main() {
  group('Identify screen', () {
    testWidgets('shows the two photo buttons once the model loads', (tester) async {
      await tester.pumpWidget(withServices(const IdentifyScreen()));
      await tester.pumpAndSettle();
      expect(find.text('Take a photo'), findsOneWidget);
      expect(find.text('Choose from gallery'), findsOneWidget);
    });

    testWidgets('shows an error with retry when the model fails to load', (tester) async {
      await tester.pumpWidget(withServices(const IdentifyScreen(),
          loadError: Exception('model missing')));
      await tester.pumpAndSettle();
      expect(find.text('The bird model failed to load'), findsOneWidget);
      expect(find.text('Try again'), findsOneWidget);
    });

    testWidgets('explains a denied camera permission', (tester) async {
      final photos = FakePhotoSource(error: const PhotoPermissionDenied());
      await tester.pumpWidget(withServices(const IdentifyScreen(), photos: photos));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Take a photo'));
      await tester.pumpAndSettle();
      expect(find.text('Camera access is off'), findsOneWidget);
      expect(find.textContaining('Settings > Apps > BirdLens'), findsOneWidget);
      await tester.tap(find.text('OK'));
      await tester.pumpAndSettle();
      expect(find.text('Camera access is off'), findsNothing);
    });

    testWidgets('does nothing when the user cancels the picker', (tester) async {
      await tester.pumpWidget(withServices(const IdentifyScreen(), photos: FakePhotoSource()));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Choose from gallery'));
      await tester.pumpAndSettle();
      expect(find.byType(ResultScreen), findsNothing);
    });
  });

  group('Result screen', () {
    testWidgets('shows top-3 bars and saves to history', (tester) async {
      final history = InMemoryHistoryRepository();
      await tester.pumpWidget(withServices(ResultScreen(imageBytes: tinyPng), history: history));
      await tester.pumpAndSettle();
      expect(find.text('Top matches'), findsOneWidget);
      expect(find.text('Black-footed Albatross'), findsOneWidget);
      expect(find.text('70.0%'), findsOneWidget);
      expect(find.byType(LinearProgressIndicator), findsNWidgets(3));
      expect(find.byType(LowConfidenceBanner), findsNothing);
      // The facts row sits below the fold on the small test screen.
      await tester.scrollUntilVisible(find.text('Saved to history'), 100);
      expect(find.text('Saved to history'), findsOneWidget);
      expect((await history.all()).single.best.label, 'Black-footed Albatross');
    });

    testWidgets('warns when the top guess is below the threshold', (tester) async {
      await tester.pumpWidget(withServices(ResultScreen(imageBytes: tinyPng),
          classifier: FakeClassifier([0.3, 0.3, 0.2, 0.2])));
      await tester.pumpAndSettle();
      expect(find.text(LowConfidenceBanner.message), findsOneWidget);
      expect(find.text('Closest matches'), findsOneWidget);
    });

    testWidgets('shows a friendly message for a corrupt image', (tester) async {
      await tester.pumpWidget(withServices(ResultScreen(imageBytes: tinyPng),
          classifier: FakeClassifier(const [], error: const ImageDecodeException('bad'))));
      await tester.pumpAndSettle();
      expect(find.text('I could not read this image'), findsOneWidget);
      expect(find.text('Back'), findsOneWidget);
    });
  });

  group('History screen', () {
    ScanRecord record(String label, double confidence, int minute) => ScanRecord(
          imagePath: '/missing.jpg',
          thumbnailPath: '/missing_thumb.jpg',
          predictions: [Prediction(index: 0, label: label, confidence: confidence)],
          threshold: 0.5,
          createdAt: DateTime(2026, 10, 23, 9, minute),
        );

    testWidgets('shows an empty state with no scans', (tester) async {
      await tester.pumpWidget(withServices(const HistoryScreen()));
      await tester.pumpAndSettle();
      expect(find.text('No scans yet'), findsOneWidget);
    });

    testWidgets('lists scans newest first and deletes after confirming', (tester) async {
      final history = InMemoryHistoryRepository();
      await history.add(record('Laysan Albatross', 0.9, 1));
      await history.add(record('Crested Auklet', 0.3, 2));
      await tester.pumpWidget(withServices(const HistoryScreen(), history: history));
      await tester.pumpAndSettle();
      final tiles = tester.widgetList<ListTile>(find.byType(ListTile)).toList();
      expect((tiles.first.title as Text).data, 'Crested Auklet');
      // The low confidence scan is labelled as such.
      expect(find.textContaining('Not confident'), findsOneWidget);

      await tester.tap(find.byTooltip('Delete scan').first);
      await tester.pumpAndSettle();
      await tester.tap(find.text('Delete'));
      await tester.pumpAndSettle();
      expect(find.text('Crested Auklet'), findsNothing);
      expect((await history.all()).length, 1);
    });
  });

  testWidgets('Benchmark screen shows results after a run', (tester) async {
    Future<BenchmarkResult> fakeRunner(classifier, {onProgress}) async => const BenchmarkResult(
          modelBytes: 1024 * 1024,
          variant: 'float16',
          warmUp: Duration(milliseconds: 120),
          runs: 50,
          average: Duration(milliseconds: 20),
          p95: Duration(milliseconds: 25),
          fastest: Duration(milliseconds: 18),
        );
    await tester.pumpWidget(withServices(BenchmarkScreen(runner: fakeRunner)));
    await tester.pumpAndSettle();
    expect(find.text('12.0 MB'), findsOneWidget);
    await tester.tap(find.text('Run benchmark'));
    await tester.pumpAndSettle();
    expect(find.text('120.0 ms'), findsOneWidget);
    expect(find.text('25.0 ms'), findsOneWidget);
    expect(find.textContaining('depend on the device'), findsOneWidget);
  });

  testWidgets('About screen shows accuracy and limitations', (tester) async {
    await tester.pumpWidget(withServices(const AboutScreen()));
    await tester.pumpAndSettle();
    expect(find.textContaining('80.0%'), findsOneWidget);
    expect(find.text('Limitations'), findsOneWidget);
  });

  testWidgets('Navigation bar switches between the four screens', (tester) async {
    await tester.pumpWidget(withServices(const HomeShell()));
    await tester.pumpAndSettle();
    for (final tab in ['History', 'Benchmark', 'About', 'Identify']) {
      await tester.tap(find.descendant(of: find.byType(NavigationBar), matching: find.text(tab)));
      await tester.pumpAndSettle();
    }
    expect(find.text('Which bird is that?'), findsOneWidget);
  });
}
