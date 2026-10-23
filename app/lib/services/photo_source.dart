import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

enum PhotoOrigin { camera, gallery }

/// Thrown when the user has blocked camera access.
class PhotoPermissionDenied implements Exception {
  const PhotoPermissionDenied({required this.permanently});

  /// True when Android won't show the prompt again and only Settings can fix it.
  final bool permanently;
}

/// Gets photo bytes from the camera or gallery. Widget tests use a fake.
abstract class PhotoSource {
  /// Returns null when the user backs out without choosing a photo.
  Future<Uint8List?> pick(PhotoOrigin origin);

  Future<void> openSettings();
}

class ImagePickerPhotoSource implements PhotoSource {
  ImagePickerPhotoSource([ImagePicker? picker]) : _picker = picker ?? ImagePicker();

  final ImagePicker _picker;

  /// I ask Android for a photo no bigger than 512 px. CUB images are about
  /// 500 px on the long side, so the model sees birds at a similar scale to
  /// training, and the plain bilinear resize to 224 doesn't alias badly.
  static const maxDimension = 512.0;

  @override
  Future<Uint8List?> pick(PhotoOrigin origin) async {
    if (origin == PhotoOrigin.camera) {
      // I check the camera permission myself so I can explain a denial
      // instead of failing silently.
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        throw PhotoPermissionDenied(permanently: status.isPermanentlyDenied);
      }
    }
    try {
      final file = await _picker.pickImage(
        source: origin == PhotoOrigin.camera ? ImageSource.camera : ImageSource.gallery,
        maxWidth: maxDimension,
        maxHeight: maxDimension,
        imageQuality: 95,
      );
      return file == null ? null : await file.readAsBytes();
    } on PlatformException catch (e) {
      // image_picker reports a denied camera with this code.
      if (e.code == 'camera_access_denied') {
        throw const PhotoPermissionDenied(permanently: false);
      }
      rethrow;
    }
  }

  @override
  Future<void> openSettings() => openAppSettings();
}
