import 'dart:io';

import 'package:flutter/material.dart';

import '../app_services.dart';
import '../models/scan_record.dart';
import '../utils/formatting.dart';
import '../widgets/message_view.dart';
import 'result_screen.dart';

/// Past scans, newest first, with swipe or tap to delete.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<ScanRecord>? _records;
  Object? _error;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_records == null && _error == null) _reload();
  }

  Future<void> _reload() async {
    try {
      final records = await AppServices.of(context).history.all();
      if (mounted) setState(() => _records = records);
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  Future<bool> _confirmDelete(ScanRecord record) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete this scan?'),
        content: Text('${record.best.label} from ${formatTimestamp(record.createdAt)} '
            'and its photo will be removed from this phone.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
        ],
      ),
    );
    return ok ?? false;
  }

  Future<void> _delete(ScanRecord record) async {
    final history = AppServices.of(context).history;
    // I drop it from the list straight away so the swipe feels instant.
    setState(() => _records = [..._records!]..removeWhere((r) => r.id == record.id));
    await history.delete(record);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: _body(context),
    );
  }

  Widget _body(BuildContext context) {
    if (_error != null) {
      return MessageView(
        icon: Icons.error_outline,
        isError: true,
        title: 'Could not open your history',
        body: '$_error',
        actionLabel: 'Try again',
        onAction: () {
          setState(() => _error = null);
          _reload();
        },
      );
    }
    final records = _records;
    if (records == null) return const Center(child: CircularProgressIndicator());
    if (records.isEmpty) {
      return const MessageView(
        icon: Icons.history,
        title: 'No scans yet',
        body: 'Birds you identify will show up here.',
      );
    }
    return RefreshIndicator(
      onRefresh: _reload,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(vertical: 8),
        itemCount: records.length,
        separatorBuilder: (_, _) => const Divider(height: 1, indent: 88),
        itemBuilder: (context, i) {
          final record = records[i];
          return Dismissible(
            key: ValueKey(record.id),
            direction: DismissDirection.endToStart,
            confirmDismiss: (_) => _confirmDelete(record),
            onDismissed: (_) => _delete(record),
            background: Container(
              alignment: Alignment.centerRight,
              padding: const EdgeInsets.only(right: 24),
              color: Theme.of(context).colorScheme.errorContainer,
              child: Icon(Icons.delete_outline,
                  color: Theme.of(context).colorScheme.onErrorContainer),
            ),
            child: _HistoryTile(
              record: record,
              onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(
                builder: (_) => ResultScreen(record: record),
              )),
              onDelete: () async {
                if (await _confirmDelete(record)) await _delete(record);
              },
            ),
          );
        },
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  const _HistoryTile({required this.record, required this.onTap, required this.onDelete});

  final ScanRecord record;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return ListTile(
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      leading: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.file(
          File(record.thumbnailPath),
          width: 56,
          height: 56,
          fit: BoxFit.cover,
          // A missing thumbnail shouldn't break the whole list.
          errorBuilder: (_, _, _) => Container(
            width: 56,
            height: 56,
            color: scheme.surfaceContainerHighest,
            child: Icon(Icons.image_not_supported_outlined, color: scheme.onSurfaceVariant),
          ),
        ),
      ),
      title: Text(record.best.label, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: Text(
        '${record.isConfident ? '' : 'Not confident · '}'
        '${formatPercent(record.best.confidence)} · ${formatTimestamp(record.createdAt)}',
      ),
      trailing: IconButton(
        tooltip: 'Delete scan',
        icon: const Icon(Icons.delete_outline),
        onPressed: onDelete,
      ),
    );
  }
}
