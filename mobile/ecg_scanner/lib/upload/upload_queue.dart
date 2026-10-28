// Upload queue for a weak network (TZ §6.4, addendum §18.8): photos wait on the phone, uploads
// are retried with a growing pause, the state survives an app restart. A photo is deleted
// from the phone once the server has accepted it (the server keeps it encrypted).
// With a vault (the app always has one) the waiting photo is encrypted on disk
// (TZ addendum §18.9); a plain `.jpg` left by an older version is still sent.
// A study on several sheets (§18.4) is one item with its sheets in order; while the doctor
// is still taking them it is a draft: kept on disk, never sent until they send it.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../api/scan_api.dart';
import '../security/local_vault.dart';

enum UploadState { draft, queued, uploading, sent, failed }

class UploadItem {
  UploadItem({
    required this.localId,
    required this.createdAt,
    required this.hints,
    this.state = UploadState.queued,
    this.scanId,
    this.attempts = 0,
    this.nextTryAt,
    this.error,
    List<String>? partIds,
  }) : partIds = partIds ?? [localId];

  final String localId;

  /// The sheets' files in the order they are sent (one for a single photo).
  final List<String> partIds;
  final DateTime createdAt;
  ScanHints hints;
  UploadState state;
  String? scanId;
  int attempts;
  DateTime? nextTryAt;
  String? error;

  Map<String, dynamic> toJson() => {
        'local_id': localId,
        'created_at': createdAt.toIso8601String(),
        'hints': hints.toJson(),
        'state': state.name,
        'scan_id': scanId,
        'attempts': attempts,
        'next_try_at': nextTryAt?.toIso8601String(),
        'error': error,
        'part_ids': partIds,
      };

  factory UploadItem.fromJson(Map<String, dynamic> j) => UploadItem(
        localId: j['local_id'] as String,
        createdAt: DateTime.parse(j['created_at'] as String),
        hints: ScanHints.fromJson(j['hints'] as Map<String, dynamic>),
        // An upload cut by an app restart is simply queued again.
        state: UploadState.values.byName(j['state'] as String) == UploadState.uploading
            ? UploadState.queued
            : UploadState.values.byName(j['state'] as String),
        scanId: j['scan_id'] as String?,
        attempts: j['attempts'] as int? ?? 0,
        nextTryAt: j['next_try_at'] == null ? null : DateTime.parse(j['next_try_at'] as String),
        error: j['error'] as String?,
        partIds: (j['part_ids'] as List<dynamic>?)?.cast<String>(),
      );
}

class UploadQueue {
  UploadQueue(this.dir, this.api,
      {DateTime Function()? clock,
      this.baseDelay = const Duration(seconds: 5),
      this.maxDelay = const Duration(minutes: 5),
      this.vault,
      Future<void> Function(File file, List<int> bytes)? writeFile})
      : _clock = clock ?? DateTime.now,
        _writeFile = writeFile ?? ((f, b) => f.writeAsBytes(b, flush: true));

  final Directory dir;
  ScanApi api;
  final LocalVault? vault;
  final DateTime Function() _clock;
  final Duration baseDelay;
  final Duration maxDelay;
  final List<UploadItem> items = [];
  bool _pumping = false;
  final Future<void> Function(File file, List<int> bytes) _writeFile;

  /// Photos the phone could not write (no space left): kept in memory, sent from there, and
  /// written again at the next pump (§18.8: the doctor's work is never lost).
  final Map<String, Uint8List> _unsaved = {};

  /// Photos waiting in memory because the disk refused them.
  bool get diskFull => _unsaved.isNotEmpty;

  File get _index => File('${dir.path}/index.json');
  File _plain(String id) => File('${dir.path}/$id.jpg');
  File _sealed(String id) => File('${dir.path}/$id.bin');

  Future<void> _write(String id, Uint8List jpeg) async {
    final v = vault;
    try {
      if (v == null) {
        await _writeFile(_plain(id), jpeg);
      } else {
        await _writeFile(_sealed(id), await v.seal(jpeg));
      }
      _unsaved.remove(id);
    } on FileSystemException catch (_) {
      _unsaved[id] = jpeg;
    }
  }

  Future<void> _flushUnsaved() async {
    for (final e in _unsaved.entries.toList()) {
      await _write(e.key, e.value);
    }
  }

  Future<Uint8List> _read(String id) async {
    final kept = _unsaved[id];
    if (kept != null) return kept;
    final v = vault;
    if (v != null && await _sealed(id).exists()) return v.open(await _sealed(id).readAsBytes());
    return _plain(id).readAsBytes();
  }

  Future<void> _delete(String id) async {
    _unsaved.remove(id);
    for (final f in [_sealed(id), _plain(id)]) {
      if (await f.exists()) await f.delete();
    }
  }

  Future<void> load() async {
    await dir.create(recursive: true);
    items.clear();
    if (await _index.exists()) {
      final list = jsonDecode(await _index.readAsString()) as List<dynamic>;
      items.addAll(list.map((e) => UploadItem.fromJson(e as Map<String, dynamic>)));
    }
  }

  Future<void> _save() async {
    final tmp = File('${_index.path}.tmp');
    try {
      await _writeFile(tmp, utf8.encode(jsonEncode(items.map((e) => e.toJson()).toList())));
      await tmp.rename(_index.path); // atomic: a crash never leaves half an index
    } on FileSystemException catch (_) {
      // No space: the list lives in memory until a later save succeeds (the next pump).
    }
  }

  Future<UploadItem> add(Uint8List jpeg, ScanHints hints) async {
    final now = _clock();
    final id = _newId();
    await _write(id, jpeg);
    final item = UploadItem(localId: id, createdAt: now, hints: hints);
    items.insert(0, item);
    await _save();
    return item;
  }

  String _newId() =>
      '${_clock().microsecondsSinceEpoch}-${math.Random().nextInt(1 << 32).toRadixString(16)}';

  /// The study being taken sheet by sheet, if any (one at a time).
  UploadItem? get draft => items.where((e) => e.state == UploadState.draft).firstOrNull;

  /// A sheet added to the draft (started if there is none); on disk at once.
  Future<UploadItem> addSheet(Uint8List jpeg, ScanHints hints) async {
    var d = draft;
    if (d == null) {
      d = UploadItem(localId: _newId(), createdAt: _clock(), hints: hints, state: UploadState.draft, partIds: []);
      items.insert(0, d);
    }
    final id = d.partIds.isEmpty ? d.localId : _newId();
    await _write(id, jpeg);
    d.partIds.add(id);
    await _save();
    return d;
  }

  Future<Uint8List> sheet(UploadItem item, int index) => _read(item.partIds[index]);

  Future<void> moveSheet(UploadItem item, int from, int to) async {
    final id = item.partIds.removeAt(from);
    item.partIds.insert(to.clamp(0, item.partIds.length), id);
    await _save();
  }

  Future<void> removeSheet(UploadItem item, int index) async {
    await _delete(item.partIds.removeAt(index));
    if (item.partIds.isEmpty) items.remove(item);
    await _save();
  }

  /// The draft goes to the queue as one study (several sheets: `pages`).
  Future<void> send(UploadItem item, ScanHints hints) async {
    item.hints = ScanHints(
        paperSpeedMmS: hints.paperSpeedMmS,
        gainMmMv: hints.gainMmMv,
        headerMasked: hints.headerMasked,
        capture: hints.capture,
        device: hints.device,
        pages: item.partIds.length > 1);
    item.state = UploadState.queued;
    await _save();
  }

  Future<void> discard(UploadItem item) async {
    for (final id in item.partIds) {
      await _delete(id);
    }
    items.remove(item);
    await _save();
  }

  /// A refused upload sent again on the doctor's request (a fixed key, a new server).
  Future<void> retry(UploadItem item) async {
    if (item.state != UploadState.failed) return;
    item.state = UploadState.queued;
    item.nextTryAt = null;
    await _save();
  }

  Duration _delay(int attempts) {
    final ms = baseDelay.inMilliseconds * math.pow(2, attempts - 1);
    return Duration(milliseconds: math.min(ms.toInt(), maxDelay.inMilliseconds));
  }

  /// Upload everything that is due; returns how many were sent.
  Future<int> pump() async {
    if (_pumping) return 0;
    _pumping = true;
    var sent = 0;
    try {
      if (_unsaved.isNotEmpty) {
        await _flushUnsaved();
        await _save();
      }
      for (final item in items.where((e) => e.state == UploadState.queued).toList()) {
        if (item.nextTryAt != null && item.nextTryAt!.isAfter(_clock())) continue;
        item.state = UploadState.uploading;
        try {
          final sheets = [for (final id in item.partIds) await _read(id)];
          item.scanId = await api.submit(sheets.first, item.hints, more: sheets.sublist(1));
          item.state = UploadState.sent;
          item.error = null;
          item.nextTryAt = null;
          for (final id in item.partIds) {
            await _delete(id);
          }
          sent++;
        } on ApiError catch (e) {
          item.attempts++;
          item.error = e.message;
          if (e.transient) {
            item.state = UploadState.queued;
            item.nextTryAt = _clock().add(_delay(item.attempts));
          } else {
            item.state = UploadState.failed; // refused (bad key, bad file): retrying cannot help
          }
        }
        await _save();
      }
    } finally {
      _pumping = false;
    }
    return sent;
  }

  /// Seconds until the next retry of `item` (for "retry in N s").
  int? retryIn(UploadItem item) {
    final at = item.nextTryAt;
    if (at == null || item.state != UploadState.queued) return null;
    return math.max(0, at.difference(_clock()).inSeconds);
  }
}
