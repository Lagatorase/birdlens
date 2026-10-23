import 'dart:typed_data';

import 'package:birdlens/services/preprocessing.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// Build an RgbImage from a function of (x, y, channel).
RgbImage makeImage(int w, int h, int Function(int x, int y, int c) f) {
  final pixels = Uint8List(w * h * 3);
  var i = 0;
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      for (var c = 0; c < 3; c++) {
        pixels[i++] = f(x, y, c);
      }
    }
  }
  return RgbImage(w, h, pixels);
}

void main() {
  group('resizeBilinear', () {
    test('keeps a flat colour flat', () {
      final src = makeImage(37, 23, (x, y, c) => [10, 120, 250][c]);
      final out = resizeBilinear(src, 224, 224);
      expect(out.length, 224 * 224 * 3);
      for (var i = 0; i < out.length; i += 3) {
        expect(out[i], closeTo(10, 1e-4));
        expect(out[i + 1], closeTo(120, 1e-4));
        expect(out[i + 2], closeTo(250, 1e-4));
      }
    });

    test('same size is an exact copy', () {
      final src = makeImage(8, 5, (x, y, c) => (x * 31 + y * 17 + c * 7) % 256);
      final out = resizeBilinear(src, 5, 8);
      for (var i = 0; i < out.length; i++) {
        expect(out[i], src.pixels[i].toDouble());
      }
    });

    test('halving a 4 pixel row averages neighbours like TensorFlow', () {
      // With half-pixel centres, output x=0 samples source x=0.5 and output
      // x=1 samples source x=2.5. tf.image.resize gives [5, 25] for this row.
      final row = [0, 10, 20, 30];
      final src = makeImage(4, 1, (x, y, c) => row[x]);
      final out = resizeBilinear(src, 1, 2);
      expect(out[0], closeTo(5, 1e-6));
      expect(out[3], closeTo(25, 1e-6));
    });

    test('upscaling clamps at the edges like TensorFlow', () {
      // tf.image.resize([[0, 100]], (1, 4), bilinear) gives [0, 25, 75, 100].
      final src = makeImage(2, 1, (x, y, c) => x == 0 ? 0 : 100);
      final out = resizeBilinear(src, 1, 4);
      expect([out[0], out[3], out[6], out[9]], [0, 25, 75, 100]);
    });

    test('values stay within 0 to 255', () {
      final src = makeImage(50, 40, (x, y, c) => (x * y * (c + 1)) % 256);
      final out = resizeBilinear(src, 224, 224);
      for (final v in out) {
        expect(v, inInclusiveRange(0, 255));
      }
    });
  });

  group('decodeRgb', () {
    test('decodes a PNG into RGB bytes', () {
      final image = img.Image(width: 3, height: 2);
      img.fill(image, color: img.ColorRgb8(200, 100, 50));
      final decoded = decodeRgb(img.encodePng(image));
      expect(decoded.width, 3);
      expect(decoded.height, 2);
      expect(decoded.pixels.sublist(0, 3), [200, 100, 50]);
    });

    test('turns greyscale into three equal channels', () {
      final grey = img.Image(width: 2, height: 2, numChannels: 1);
      img.fill(grey, color: img.ColorUint8.rgb(90, 90, 90));
      final decoded = decodeRgb(img.encodePng(grey));
      expect(decoded.pixels.length, 2 * 2 * 3);
      expect(decoded.pixels.toSet(), {90});
    });

    test('drops the alpha channel', () {
      final rgba = img.Image(width: 1, height: 1, numChannels: 4);
      rgba.setPixelRgba(0, 0, 10, 20, 30, 128);
      expect(decodeRgb(img.encodePng(rgba)).pixels, [10, 20, 30]);
    });

    test('throws a clear error for bytes that are not an image', () {
      expect(() => decodeRgb(Uint8List.fromList(List.filled(64, 7))),
          throwsA(isA<ImageDecodeException>()));
      expect(() => decodeRgb(Uint8List(0)), throwsA(isA<ImageDecodeException>()));
    });
  });

  group('quantize', () {
    test('uint8 with scale 1 and zero point 0 just rounds', () {
      final q = quantize(Float32List.fromList([0, 0.4, 0.5, 254.6, 300]), 1.0, 0, signed: false);
      expect(q, [0, 0, 1, 255, 255]);
    });

    test('int8 applies the zero point and clamps', () {
      final q = quantize(Float32List.fromList([0, 128, 255]), 1.0, -128, signed: true);
      expect(q, [-128, 0, 127]);
    });
  });

  test('preprocessBytes gives a 224x224x3 tensor', () {
    final image = img.Image(width: 64, height: 48);
    img.fill(image, color: img.ColorRgb8(1, 2, 3));
    final out = preprocessBytes(img.encodePng(image), 224);
    expect(out.length, 224 * 224 * 3);
    expect(out.sublist(0, 3), [1, 2, 3]);
  });
}
