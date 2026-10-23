import 'package:flutter/material.dart';

import 'about_screen.dart';
import 'benchmark_screen.dart';
import 'history_screen.dart';
import 'identify_screen.dart';

/// Bottom navigation between the four main screens.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;

  // I bump this each time the History tab opens so it reloads new scans.
  int _historyVisit = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      const IdentifyScreen(),
      HistoryScreen(key: ValueKey(_historyVisit)),
      const BenchmarkScreen(),
      const AboutScreen(),
    ];
    return Scaffold(
      body: pages[_index],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() {
          if (i == 1) _historyVisit++;
          _index = i;
        }),
        destinations: const [
          NavigationDestination(
              icon: Icon(Icons.center_focus_weak_outlined),
              selectedIcon: Icon(Icons.center_focus_strong),
              label: 'Identify'),
          NavigationDestination(
              icon: Icon(Icons.history_outlined),
              selectedIcon: Icon(Icons.history),
              label: 'History'),
          NavigationDestination(
              icon: Icon(Icons.speed_outlined),
              selectedIcon: Icon(Icons.speed),
              label: 'Benchmark'),
          NavigationDestination(
              icon: Icon(Icons.info_outline),
              selectedIcon: Icon(Icons.info),
              label: 'About'),
        ],
      ),
    );
  }
}
