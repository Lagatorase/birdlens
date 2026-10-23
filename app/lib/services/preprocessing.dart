import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Thrown when the picked file can't be decoded as an image.
class ImageDecodeException implements Exception {
  const ImageDecodeException(this.message);
  final String message;

  @override
  String toString() => 'ImageDecodeException: $message';
}

/// Raw RGB pixels, row by row, three bytes per pixel.
class RgbImage {
  const RgbImage(this.width, this.height, this.pixels);
  final int width;
  final int height;
  final Uint8List pixels;
}

/// Decode JPEG/PNG bytes into plain RGB pixels with the EXIF rotation applied.
RgbImage decodeRgb(Uint8List bytes) {
  // I let the image package detect the format from the bytes.
  final decoded = img.decodeImage(bytes);
  if (decoded == null || decoded.width == 0 || decoded.height == 0) {
    throw const ImageDecodeException('This file is not an image I can read.');
  }
  // Phone photos often store rotation in EXIF, so I bake it into the pixels.
  final oriented = img.bakeOrientation(decoded);
  // I force 8-bit RGB, which also drops alpha and expands greyscale,
  // the same as decode_image(channels=3) does in Python.
  final rgb = oriented.convert(format: img.Format.uint8, numChannels: 3);
  final pixels = rgb.getBytes(order: img.ChannelOrder.rgb);
  return RgbImage(rgb.width, rgb.height, pixels);
}

/// Bilinear resize that matches tf.image.resize(method="bilinear", antialias=False).
///
/// The output is HWC float32 in the 0 to 255 range, which is exactly what the
/// model's input expects, because the model rescales internally.
Float32List resizeBilinear(RgbImage src, int outHeight, int outWidth) {
  final out = Float32List(outHeight * outWidth * 3);
  final inH = src.height;
  final inW = src.width;
  final p = src.pixels;
  // These are the scale factors TensorFlow uses with half-pixel centres.
  final scaleY = inH / outHeight;
  final scaleX = inW / outWidth;

  // I precompute the x lookups once, because they repeat on every row.
  final x0s = Int32List(outWidth);
  final x1s = Int32List(outWidth);
  final dxs = Float64List(outWidth);
  for (var x = 0; x < outWidth; x++) {
    // Half-pixel centres: the centre of output pixel x maps to this source x.
    // TensorFlow clamps negative positions to 0 at the left edge.
    final sx = ((x + 0.5) * scaleX - 0.5).clamp(0.0, double.infinity);
    final x0 = sx.floor().clamp(0, inW - 1);
    x0s[x] = x0;
    x1s[x] = (x0 + 1).clamp(0, inW - 1);
    dxs[x] = sx - x0;
  }

  var o = 0;
  for (var y = 0; y < outHeight; y++) {
    // Same mapping for rows.
    final sy = ((y + 0.5) * scaleY - 0.5).clamp(0.0, double.infinity);
    final y0 = sy.floor().clamp(0, inH - 1);
    final y1 = (y0 + 1).clamp(0, inH - 1);
    final dy = sy - y0;
    final row0 = y0 * inW * 3;
    final row1 = y1 * inW * 3;
    for (var x = 0; x < outWidth; x++) {
      final a = row0 + x0s[x] * 3;
      final b = row0 + x1s[x] * 3;
      final c = row1 + x0s[x] * 3;
      final d = row1 + x1s[x] * 3;
      final dx = dxs[x];
      for (var ch = 0; ch < 3; ch++) {
        // I blend horizontally on both rows, then vertically between them.
        final top = p[a + ch] + (p[b + ch] - p[a + ch]) * dx;
        final bottom = p[c + ch] + (p[d + ch] - p[c + ch]) * dx;
        out[o++] = top + (bottom - top) * dy;
      }
    }
  }
  return out;
}

/// Quantise float pixels for a model with an integer input tensor.
///
/// q = round(x / scale) + zeroPoint, clamped to the integer type's range.
List<int> quantize(Float32List values, double scale, int zeroPoint,
    {required bool signed}) {
  final lo = signed ? -128 : 0;
  final hi = signed ? 127 : 255;
  final out = signed ? Int8List(values.length) : Uint8List(values.length);
  for (var i = 0; i < values.length; i++) {
    // Dart's round() goes half away from zero, and my Python export code
    // rounds the same way so the int8 parity check compares like with like.
    final q = (values[i] / scale).round() + zeroPoint;
    out[i] = q < lo ? lo : (q > hi ? hi : q);
  }
  return out;
}

/// The full app preprocessing: decode, orient, and resize to the model size.
Float32List preprocessBytes(Uint8List bytes, int size) {
  return resizeBilinear(decodeRgb(bytes), size, size);
}
