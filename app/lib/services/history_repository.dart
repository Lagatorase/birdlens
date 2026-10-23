import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../models/scan_record.dart';

/// Storage for past scans. Widget tests use the in-memory version.
abstract class HistoryRepository {
  Future<List<ScanRecord>> all();
  Future<ScanRecord> add(ScanRecord record);
  Future<void> delete(ScanRecord record);
}

/// History kept in a local SQLite database with sqflite.
class SqfliteHistoryRepository implements HistoryRepository {
  SqfliteHistoryRepository._(this._db);

  final Database _db;
  static const _table = 'scans';

  static Future<SqfliteHistoryRepository> open() async {
    final dir = await getDatabasesPath();
    final db = await openDatabase(
      p.join(dir, 'birdlens.db'),
      version: 1,
      onCreate: (db, version) => db.execute('''
        CREATE TABLE $_table (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          image_path TEXT NOT NULL,
          thumbnail_path TEXT NOT NULL,
          top_label TEXT NOT NULL,
          top_confidence REAL NOT NULL,
          predictions_json TEXT NOT NULL,
          threshold REAL NOT NULL,
          created_at INTEGER NOT NULL
        )
      '''),
    );
    return SqfliteHistoryRepository._(db);
  }

  @override
  Future<List<ScanRecord>> all() async {
    // Newest scans first, which is what people look for.
    final rows = await _db.query(_table, orderBy: 'created_at DESC');
    return rows.map(ScanRecord.fromRow).toList();
  }

  @override
  Future<ScanRecord> add(ScanRecord record) async {
    final id = await _db.insert(_table, record.toRow());
    return record.copyWith(id: id);
  }

  @override
  Future<void> delete(ScanRecord record) async {
    await _db.delete(_table, where: 'id = ?', whereArgs: [record.id]);
    // I also remove the saved photo and thumbnail so deleted scans don't
    // keep using storage.
    for (final path in [record.imagePath, record.thumbnailPath]) {
      final file = File(path);
      if (await file.exists()) await file.delete();
    }
  }
}

/// A simple in-memory history for tests.
class InMemoryHistoryRepository implements HistoryRepository {
  final List<ScanRecord> _records = [];
  int _nextId = 1;

  @override
  Future<List<ScanRecord>> all() async =>
      [..._records]..sort((a, b) => b.createdAt.compareTo(a.createdAt));

  @override
  Future<ScanRecord> add(ScanRecord record) async {
    final saved = record.copyWith(id: _nextId++);
    _records.add(saved);
    return saved;
  }

  @override
  Future<void> delete(ScanRecord record) async {
    _records.removeWhere((r) => r.id == record.id);
  }
}
