// Against a running /scans service (skipped unless ECGSCAN_LIVE_URL is set):
// ECGSCAN_LIVE_URL=http://127.0.0.1:8765 ECGSCAN_LIVE_KEY=... ECGSCAN_LIVE_PHOTO=page.jpg flutter test test/live_server_test.dart
import 'dart:io';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/masking/header_mask.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final url = Platform.environment['ECGSCAN_LIVE_URL'];
  test('upload, wait, confirm, analyse against the real service', () async {
    final api = ScanApi(baseUrl: url!, apiKey: Platform.environment['ECGSCAN_LIVE_KEY']!);
    final cfg = await api.captureQcConfig();
    expect((cfg['config'] as Map)['thresholds'], isNotNull);
    final photo = maskHeader(File(Platform.environment['ECGSCAN_LIVE_PHOTO']!).readAsBytesSync(), bandFraction: 0.14);
    final id = await api.submit(photo, const ScanHints(paperSpeedMmS: 25, gainMmMv: 10, headerMasked: true));
    Map<String, dynamic> scan;
    final end = DateTime.now().add(const Duration(seconds: 120));
    do {
      await Future<void>.delayed(const Duration(seconds: 1));
      scan = await api.get(id);
    } while (const ['queued', 'processing'].contains(scan['status']) && DateTime.now().isBefore(end));
    final result = scan['result'] as Map<String, dynamic>;
    expect((result['source'] as Map)['header_masked'], isTrue);
    expect((result['source'] as Map)['client'], 'flutter');
    expect((result['calibration'] as Map)['speed_source'], 'user');
    expect((await api.overlay(id)).length, greaterThan(1000));
    if (scan['status'] == 'needs_confirmation') await api.confirm(id, 'accepted');
    final a = await api.analyze(id);
    expect(a['conclusion_ru'] as String, contains('Получено по фото'));
    stdout.writeln('live: ${scan['status']} -> ${a['mode']}');
  }, skip: url == null ? 'ECGSCAN_LIVE_URL not set' : false, timeout: const Timeout(Duration(minutes: 3)));
}
