// Upload queue for a weak network (TZ §6.4): photos wait on the phone, uploads are
// retried with a growing pause, the state survives an app restart. A photo is deleted
// from the phone once the server has accepted it (the server keeps it encrypted).
// With a vault (the app always has one) the waiting photo is encrypted on disk
// (TZ addendum §18.9); a plain `.jpg` left by an older version is still sent.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import '../api/scan_api.dart';
import '../security/local_vault.dart';

enum UploadState { queued, uploading, sent, failed }

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
  });

  final String localId;
  final DateTime createdAt;
  final ScanHints hints;
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
      );
}

class UploadQueue {
  UploadQueue(this.dir, this.api,
      {DateTime Function()? clock,
      this.baseDelay = const Duration(seconds: 5),
      this.maxDelay = const Duration(minutes: 5),
      this.vault})
      : _clock = clock ?? DateTime.now;

  final Directory dir;
  ScanApi api;
  final LocalVault? vault;
  final DateTime Function() _clock;
  final Duration baseDelay;
  final Duration maxDelay;
  final List<UploadItem> items = [];
  bool _pumping = false;

  File get _index => File('${dir.path}/index.json');
  File _plain(String id) => File('${dir.path}/$id.jpg');
  File _sealed(String id) => File('${dir.path}/$id.bin');

  Future<void> _write(String id, Uint8List jpeg) async {
    final v = vault;
    if (v == null) return _plain(id).writeAsBytes(jpeg, flush: true).then((_) {});
    await _sealed(id).writeAsBytes(await v.seal(jpeg), flush: true);
  }

  Future<Uint8List> _read(String id) async {
    final v = vault;
    if (v != null && await _sealed(id).exists()) return v.open(await _sealed(id).readAsBytes());
    return _plain(id).readAsBytes();
  }

  Future<void> _delete(String id) async {
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
    await tmp.writeAsString(jsonEncode(items.map((e) => e.toJson()).toList()));
    await tmp.rename(_index.path); // atomic: a crash never leaves half an index
  }

  Future<UploadItem> add(Uint8List jpeg, ScanHints hints) async {
    final now = _clock();
    final id = '${now.microsecondsSinceEpoch}-${math.Random().nextInt(1 << 32).toRadixString(16)}';
    await _write(id, jpeg);
    final item = UploadItem(localId: id, createdAt: now, hints: hints);
    items.insert(0, item);
    await _save();
    return item;
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
      for (final item in items.where((e) => e.state == UploadState.queued).toList()) {
        if (item.nextTryAt != null && item.nextTryAt!.isAfter(_clock())) continue;
        item.state = UploadState.uploading;
        try {
          item.scanId = await api.submit(await _read(item.localId), item.hints);
          item.state = UploadState.sent;
          item.error = null;
          item.nextTryAt = null;
          await _delete(item.localId);
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
