import 'package:flutter/material.dart';

import 'screens/home_shell.dart';

/// The MaterialApp with BirdLens's light and dark Material 3 themes.
class BirdLensApp extends StatelessWidget {
  const BirdLensApp({super.key});

  // A muted forest green felt right for a birding app.
  static const _seed = Color(0xFF2E6B4F);

  static ThemeData theme(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(seedColor: _seed, brightness: brightness);
    return ThemeData(
      colorScheme: scheme,
      useMaterial3: true,
      cardTheme: CardThemeData(
        elevation: 0,
        color: scheme.surfaceContainerLow,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
      appBarTheme: const AppBarTheme(centerTitle: false),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'BirdLens',
      debugShowCheckedModeBanner: false,
      theme: theme(Brightness.light),
      darkTheme: theme(Brightness.dark),
      home: const HomeShell(),
    );
  }
}
