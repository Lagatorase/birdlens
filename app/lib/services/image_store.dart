import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Where a saved scan's photo and thumbnail ended up.
class StoredImage {
  const StoredImage(this.imagePath, this.thumbnailPath);
  final String imagePath;
  final String thumbnailPath;
}

/// Saves scan photos into the app's private documents folder.
abstract class ImageStore {
  Future<StoredImage> save(Uint8List bytes);
}

class FileImageStore implements ImageStore {
  FileImageStore([this._baseDir]);

  final Directory? _baseDir;

  static const thumbnailSize = 160;

  @override
  Future<StoredImage> save(Uint8List bytes) async {
    final base = _baseDir ?? await getApplicationDocumentsDirectory();
    final dir = Directory(p.join(base.path, 'scans'));
    await dir.create(recursive: true);
    // A timestamp name is unique enough for one person's scans.
    final stamp = DateTime.now().microsecondsSinceEpoch;
    final imagePath = p.join(dir.path, 'scan_$stamp.jpg');
    final thumbPath = p.join(dir.path, 'scan_${stamp}_thumb.jpg');
    // I make the thumbnail in a background isolate so the list stays smooth.
    final thumb = await Isolate.run(() => makeThumbnail(bytes, thumbnailSize));
    await File(imagePath).writeAsBytes(bytes, flush: true);
    await File(thumbPath).writeAsBytes(thumb, flush: true);
    return StoredImage(imagePath, thumbPath);
  }
}

/// A small square JPEG thumbnail with EXIF rotation applied.
Uint8List makeThumbnail(Uint8List bytes, int size) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) return bytes;
  final oriented = img.bakeOrientation(decoded);
  // copyResizeCropSquare fills the square without stretching the bird.
  final square = img.copyResizeCropSquare(oriented, size: size);
  return img.encodeJpg(square, quality: 85);
}
