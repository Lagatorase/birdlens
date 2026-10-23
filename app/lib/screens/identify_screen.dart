import 'package:flutter/material.dart';

import '../app_services.dart';
import '../services/photo_source.dart';
import '../widgets/message_view.dart';
import 'result_screen.dart';

/// The home tab: take or pick a photo to identify.
class IdentifyScreen extends StatelessWidget {
  const IdentifyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final services = AppServices.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('BirdLens')),
      // I rebuild whenever the model finishes loading or fails.
      body: ListenableBuilder(
        listenable: services.model,
        builder: (context, _) {
          switch (services.model.status) {
            case ModelStatus.loading:
              return const Center(child: CircularProgressIndicator());
            case ModelStatus.failed:
              return MessageView(
                icon: Icons.error_outline,
                isError: true,
                title: 'The bird model failed to load',
                body: services.model.error ?? 'Unknown error.',
                actionLabel: 'Try again',
                onAction: services.model.load,
              );
            case ModelStatus.ready:
              return const _PickPhoto();
          }
        },
      ),
    );
  }
}

class _PickPhoto extends StatelessWidget {
  const _PickPhoto();

  Future<void> _pick(BuildContext context, PhotoOrigin origin) async {
    final services = AppServices.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    try {
      final bytes = await services.photos.pick(origin);
      // Null means the user backed out of the picker, which is fine.
      if (bytes == null) return;
      await navigator.push(MaterialPageRoute<void>(
        builder: (_) => ResultScreen(imageBytes: bytes),
      ));
    } on PhotoPermissionDenied catch (e) {
      if (!context.mounted) return;
      await _explainDenied(context, e, services.photos);
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not open that photo. ($e)')),
      );
    }
  }

  Future<void> _explainDenied(
      BuildContext context, PhotoPermissionDenied e, PhotoSource photos) {
    return showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        icon: const Icon(Icons.no_photography_outlined),
        title: const Text('Camera access is off'),
        content: Text(e.permanently
            ? 'BirdLens needs the camera to take a photo. You can turn it on in '
                'Settings, or choose a photo from your gallery instead.'
            : 'BirdLens needs the camera to take a photo. You can allow it next '
                'time, or choose a photo from your gallery instead.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Not now'),
          ),
          if (e.permanently)
            FilledButton(
              onPressed: () {
                Navigator.pop(context);
                photos.openSettings();
              },
              child: const Text('Open settings'),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = Theme.of(context).textTheme;
    return SafeArea(
      child: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // A simple badge instead of an illustration keeps the APK small.
                Center(
                  child: Container(
                    width: 112,
                    height: 112,
                    decoration: BoxDecoration(
                      color: scheme.primaryContainer,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(Icons.flutter_dash,
                        size: 64, color: scheme.onPrimaryContainer),
                  ),
                ),
                const SizedBox(height: 24),
                Text('Which bird is that?',
                    style: text.headlineSmall, textAlign: TextAlign.center),
                const SizedBox(height: 8),
                Text(
                  'Take a photo or pick one from your gallery. '
                  'Everything runs on your phone, with no upload.',
                  style: text.bodyMedium?.copyWith(color: scheme.onSurfaceVariant),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),
                FilledButton.icon(
                  onPressed: () => _pick(context, PhotoOrigin.camera),
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Take a photo'),
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: () => _pick(context, PhotoOrigin.gallery),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Choose from gallery'),
                  style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(52)),
                ),
                const SizedBox(height: 24),
                Text(
                  'Knows 200 bird species, mostly from North America.',
                  style: text.bodySmall?.copyWith(color: scheme.onSurfaceVariant),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
