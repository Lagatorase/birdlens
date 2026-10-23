import 'package:birdlens/models/prediction.dart';
import 'package:birdlens/models/scan_record.dart';
import 'package:birdlens/services/benchmark.dart';
import 'package:birdlens/services/result_parser.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const labels = ['A', 'B', 'C', 'D', 'E'];

  group('topK', () {
    test('returns the highest scores first with their labels', () {
      final top = topK([0.1, 0.5, 0.05, 0.3, 0.05], labels, 3);
      expect(top.map((p) => p.label), ['B', 'D', 'A']);
      expect(top.map((p) => p.index), [1, 3, 0]);
      expect(top.first.confidence, 0.5);
    });

    test('breaks ties by class index so the order is stable', () {
      final top = topK([0.2, 0.2, 0.2, 0.2, 0.2], labels, 3);
      expect(top.map((p) => p.index), [0, 1, 2]);
    });

    test('clamps scores into 0 to 1', () {
      final top = topK([1.0001, -0.001, 0, 0, 0], labels, 2);
      expect(top.first.confidence, 1.0);
      expect(top.every((p) => p.confidence >= 0), isTrue);
    });

    test('rejects a label count that does not match the model', () {
      expect(() => topK([0.5, 0.5], labels, 1), throwsArgumentError);
    });
  });

  test('dequantize applies scale and zero point', () {
    // A uint8 softmax output usually has scale 1/256 and zero point 0.
    expect(dequantize([0, 128, 255], 1 / 256, 0), [0, 0.5, 255 / 256]);
    // int8 outputs use zero point -128.
    expect(dequantize([-128, 127], 1 / 256, -128), [0, 255 / 256]);
  });

  group('parseResult', () {
    test('marks a result confident at or above the threshold', () {
      final r = parseResult([0.6, 0.2, 0.1, 0.05, 0.05], labels, threshold: 0.6);
      expect(r.top.length, 3);
      expect(r.best.label, 'A');
      expect(r.isConfident, isTrue);
    });

    test('marks a result not confident below the threshold', () {
      final r = parseResult([0.3, 0.25, 0.2, 0.15, 0.1], labels, threshold: 0.6);
      expect(r.isConfident, isFalse);
    });
  });

  test('parseLabels skips blank lines and trims', () {
    expect(parseLabels('Black-footed Albatross\r\n Laysan Albatross\n\n'),
        ['Black-footed Albatross', 'Laysan Albatross']);
  });

  test('ScanRecord survives a round trip through a database row', () {
    final record = ScanRecord(
      id: 3,
      imagePath: '/a.jpg',
      thumbnailPath: '/a_thumb.jpg',
      predictions: const [
        Prediction(index: 1, label: 'B', confidence: 0.4),
        Prediction(index: 0, label: 'A', confidence: 0.3),
      ],
      threshold: 0.5,
      createdAt: DateTime(2026, 10, 23, 14, 5),
    );
    final back = ScanRecord.fromRow(record.toRow());
    expect(back.id, 3);
    expect(back.best.label, 'B');
    expect(back.predictions.length, 2);
    expect(back.isConfident, isFalse);
    expect(back.createdAt, record.createdAt);
  });

  group('benchmark summarise', () {
    test('average, p95 and fastest', () {
      final times = [for (var i = 1; i <= 20; i++) Duration(milliseconds: i)];
      final s = summarise(times);
      expect(s.average, const Duration(microseconds: 10500));
      // 95 percent of 20 runs is 19, so p95 is the 19th fastest.
      expect(s.p95, const Duration(milliseconds: 19));
      expect(s.fastest, const Duration(milliseconds: 1));
    });

    test('a single run is its own p95', () {
      final s = summarise([const Duration(milliseconds: 7)]);
      expect(s.p95, const Duration(milliseconds: 7));
    });
  });
}
