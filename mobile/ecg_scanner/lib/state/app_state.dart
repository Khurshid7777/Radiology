// App-wide state: settings, strings, the API, the upload queue and scan statuses.
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../api/scan_api.dart';
import '../i18n/strings.dart';
import '../masking/header_mask.dart';
import '../security/local_vault.dart';
import '../upload/upload_queue.dart';
import 'settings.dart';

class AppState extends ChangeNotifier {
  AppState({required this.dataDir, required this.settings, required this.strings, ScanApi? api, KeyStore? keys})
      : api = api ?? ScanApi(baseUrl: settings.serverUrl, apiKey: settings.apiKey) {
    queue = UploadQueue(Directory('${dataDir.path}/queue'), this.api, vault: LocalVault(keys ?? MemoryKeyStore()));
  }

  final Directory dataDir;
  Settings settings;
  Strings strings;
  ScanApi api;
  late final UploadQueue queue;

  /// The frame-check config from the server (null until fetched: manual shot only).
  Map<String, dynamic>? qcConfig;

  /// Last GET /scans/{id} per scan id.
  final Map<String, Map<String, dynamic>> scans = {};
  Timer? _timer;

  File get _settingsFile => File('${dataDir.path}/settings.json');

  /// `keys`: the platform keystore in the app (main.dart); tests pass none (memory).
  static Future<AppState> open(Directory dataDir, {ScanApi? api, KeyStore? keys}) async {
    final settings = await Settings.load(File('${dataDir.path}/settings.json'));
    final s = AppState(
        dataDir: dataDir, settings: settings, strings: await Strings.load(settings.lang), api: api, keys: keys);
    await s.queue.load();
    return s;
  }

  String t(String key, [Map<String, Object> args = const {}]) => strings.t(key, args);

  Future<void> saveSettings(Settings next) async {
    final langChanged = next.lang != settings.lang;
    settings = next;
    await settings.save(_settingsFile);
    if (langChanged) strings = await Strings.load(settings.lang);
    api = ScanApi(baseUrl: settings.serverUrl, apiKey: settings.apiKey);
    queue.api = api;
    qcConfig = null;
    notifyListeners();
  }

  /// Background work: uploads, statuses, the frame-check config.
  void start({Duration every = const Duration(seconds: 5)}) {
    _timer?.cancel();
    _timer = Timer.periodic(every, (_) => tick());
    tick();
  }

  Future<void> tick() async {
    if (!settings.configured) return;
    if (qcConfig == null) {
      try {
        qcConfig = (await api.captureQcConfig())['config'] as Map<String, dynamic>;
      } on ApiError catch (_) {}
    }
    await queue.pump();
    for (final item in queue.items.where((e) => e.state == UploadState.sent && e.scanId != null)) {
      final last = scans[item.scanId!];
      if (last != null && !const ['queued', 'processing'].contains(last['status'])) continue;
      await refresh(item.scanId!);
    }
    notifyListeners();
  }

  Future<Map<String, dynamic>?> refresh(String scanId) async {
    try {
      scans[scanId] = await api.get(scanId);
      notifyListeners();
      return scans[scanId];
    } on ApiError catch (_) {
      return scans[scanId];
    }
  }

  /// A taken photo: header masked when set (TZ §6.2), queued, upload attempted. Unmasked,
  /// the bytes are the camera's own, sent as they are (TZ addendum §18.1); `capture` is the
  /// capture record (§18.13) the masking method is added to.
  Future<void> addPhoto(Uint8List jpeg, {double? speed, double? gain, double? band, Map<String, dynamic>? capture}) async {
    final masked = settings.maskHeader && (band ?? settings.headerBand) > 0;
    final bytes = masked ? await compute(_mask, (jpeg, band ?? settings.headerBand)) : jpeg;
    final record = capture == null ? null : {...capture, 'mask_method': masked ? 'template_zone' : 'none'};
    await queue.add(bytes, ScanHints(paperSpeedMmS: speed, gainMmMv: gain, headerMasked: masked, capture: record));
    notifyListeners();
    unawaited(tick());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

Uint8List _mask((Uint8List, double) a) => maskHeader(a.$1, bandFraction: a.$2);
