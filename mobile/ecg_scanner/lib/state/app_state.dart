// App-wide state: settings, strings, the API, the upload queue and scan statuses.
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../api/scan_api.dart';
import '../i18n/strings.dart';
import '../masking/header_mask.dart';
import '../masking/zone_mask.dart';
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

  /// The photo as it leaves the phone (TZ §6.2, addendum §18.6; S3): with the recorder's profile
  /// and the confirmed sheet corners, its personal-data zones are painted white (`band` > 0 adds
  /// the doctor's top band on top: the manual correction); without a profile, the top band.
  /// Unmasked, the bytes are the camera's own (TZ addendum §18.1).
  Future<(Uint8List, bool, String)> _prepare(Uint8List jpeg, double? band, Map<String, dynamic>? capture) async {
    if (!settings.maskHeader) return (jpeg, false, 'none');
    final corners = (capture?['corners'] as Map?)?['final'] as List?;
    if (settings.hasZones && corners != null && corners.length == 4) {
      final quad = [for (final c in corners) [((c as List)[0] as num).toDouble(), (c[1] as num).toDouble()]];
      final extra = band ?? 0;
      final bytes = await compute(_maskZones, (jpeg, settings.deviceZones, quad,
          (capture?['rotate_deg'] as int?) ?? 0, settings.zoneMargin, extra));
      return (bytes, true, extra > 0 ? 'manual' : 'profile_zones');
    }
    final b = band ?? settings.headerBand;
    if (b <= 0) return (jpeg, false, 'none');
    return (await compute(_mask, (jpeg, b)), true, 'template_zone');
  }

  /// A taken photo: masked (see `_prepare`), queued, upload attempted; `capture` is the capture
  /// record (§18.13) the masking method is added to.
  Future<void> addPhoto(Uint8List jpeg, {double? speed, double? gain, double? band, Map<String, dynamic>? capture}) async {
    final (bytes, masked, method) = await _prepare(jpeg, band, capture);
    final record = capture == null ? null : {...capture, 'mask_method': method};
    await queue.add(bytes, ScanHints(paperSpeedMmS: speed, gainMmMv: gain, headerMasked: masked, capture: record,
        deviceProfile: settings.deviceProfileId));
    notifyListeners();
    unawaited(tick());
  }

  /// A sheet of a study on several pages (TZ addendum §18.4): masked like a single photo and
  /// kept in the draft on disk until the doctor sends the study.
  Future<void> addSheet(Uint8List jpeg, {double? band, Map<String, dynamic>? capture}) async {
    final (bytes, masked, method) = await _prepare(jpeg, band, capture);
    final record = capture == null ? null : {...capture, 'mask_method': method};
    await queue.addSheet(bytes, ScanHints(headerMasked: masked, capture: record, deviceProfile: settings.deviceProfileId));
    notifyListeners();
  }

  /// The draft goes to the queue as one study with the speed and gain the doctor chose.
  Future<void> sendDraft({double? speed, double? gain}) async {
    final d = queue.draft;
    if (d == null) return;
    await queue.send(d, ScanHints(paperSpeedMmS: speed, gainMmMv: gain, headerMasked: d.hints.headerMasked,
        capture: d.hints.capture, deviceProfile: d.hints.deviceProfile));
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

Uint8List _maskZones((Uint8List, List<List<double>>, List<List<double>>, int, double, double) a) =>
    maskZones(a.$1, a.$2, a.$3, a.$4, margin: a.$5, extraBand: a.$6);
