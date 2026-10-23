/// Small text helpers so I don't need the intl package.
library;

String formatPercent(double value) => '${(value * 100).toStringAsFixed(1)}%';

/// Decimal units (1 MB = 1,000,000 bytes), the same as the README tables.
String formatBytes(int bytes) {
  if (bytes < 1000) return '$bytes B';
  if (bytes < 1000 * 1000) return '${(bytes / 1000).toStringAsFixed(1)} KB';
  return '${(bytes / (1000 * 1000)).toStringAsFixed(1)} MB';
}

String formatMs(Duration d) => '${(d.inMicroseconds / 1000).toStringAsFixed(1)} ms';

const _months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/// For example "23 Oct, 14:05", with the year added only for older scans
/// so the history list doesn't wrap.
String formatTimestamp(DateTime t, {DateTime? now}) {
  String two(int n) => n.toString().padLeft(2, '0');
  final year = t.year == (now ?? DateTime.now()).year ? '' : ' ${t.year}';
  return '${t.day} ${_months[t.month - 1]}$year, ${two(t.hour)}:${two(t.minute)}';
}
