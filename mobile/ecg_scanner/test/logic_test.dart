import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/i18n/strings.dart';
import 'package:ecg_scanner/masking/header_mask.dart';
import 'package:ecg_scanner/qc/auto_capture.dart';
import 'package:ecg_scanner/qc/capture_qc.dart';
import 'package:ecg_scanner/upload/upload_queue.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;

FrameQc qc(List<String> hints) => FrameQc(
    width: 1, height: 1, factor: 1, brightness: 200, glare: 0, sharpness: 100, tiltDeg: 0, periodPx: 20,
    pxPerMm: 12, sheet: 0.7, hints: hints);

Uint8List jpeg(int w, int h, int grey) =>
    Uint8List.fromList(img.encodeJpg(img.Image(width: w, height: h)..clear(img.ColorRgb8(grey, grey, grey))));

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('strings', () {
    test('both languages, placeholders, no silent fallback', () async {
      final ru = await Strings.load('ru'), uz = await Strings.load('uz');
      expect(ru.t('app.retry_in', {'seconds': 40}), 'Нет связи, повтор через 40 с');
      expect(uz.t('qc.closer'), startsWith('Yaqinroq'));
      expect(() => ru.t('no.such.key'), throwsArgumentError);
    });
  });

  group('auto-capture', () {
    test('fires once after N good frames; a bad frame resets the streak', () {
      final a = AutoCapture(3);
      expect(a.update(qc(['glare', 'frame'])).hint, 'glare'); // priority order, not list order
      expect(a.update(qc([])).fire, isFalse);
      expect(a.update(qc([])).fire, isFalse);
      expect(a.update(qc(['steady'])).streak, 0);
      a.update(qc([]));
      a.update(qc([]));
      final s = a.update(qc([]));
      expect(s.fire, isTrue);
      expect(a.update(qc([])).fire, isFalse); // once
    });
  });

  group('header mask', () {
    test('the top band is white, the rest untouched, no EXIF', () {
      final out = img.decodeJpg(maskHeader(jpeg(200, 100, 40), bandFraction: 0.2))!;
      expect(out.getPixel(100, 5).r, greaterThan(245));
      expect(out.getPixel(100, 18).r, greaterThan(245));
      expect(out.getPixel(100, 60).r, lessThan(60));
      expect(out.exif.isEmpty, isTrue);
    });
  });

  group('api', () {
    test('submit sends the file, hints and key; errors are typed', () async {
      late http.BaseRequest seen;
      final client = MockClient.streaming((req, body) async {
        seen = req;
        final text = await latin1.decodeStream(body);
        expect(text, contains('name="paper_speed_mm_s"'));
        expect(text, contains('name="client"\r\n\r\nflutter'));
        return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s1", "status": "queued"}')), 202);
      });
      final api = ScanApi(baseUrl: 'http://srv/', apiKey: 'k', client: client);
      expect(await api.submit(jpeg(10, 10, 200), const ScanHints(paperSpeedMmS: 25, headerMasked: true)), 's1');
      expect(seen.url.toString(), 'http://srv/scans');
      expect(seen.headers['X-API-Key'], 'k');

      final bad = ScanApi(baseUrl: 'http://srv', apiKey: 'k',
          client: MockClient((r) async => http.Response('{"detail": "invalid or missing X-API-Key"}', 401)));
      await expectLater(bad.get('x'), throwsA(isA<ApiError>().having((e) => e.transient, 'transient', isFalse)));
      final down = ScanApi(baseUrl: 'http://srv', apiKey: 'k',
          client: MockClient((r) async => throw const SocketException('no route')));
      await expectLater(down.get('x'), throwsA(isA<ApiError>().having((e) => e.transient, 'transient', isTrue)));
    });

    test('confirm posts the decision and corrections', () async {
      final api = ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: MockClient((r) async {
        expect(r.url.path, '/scans/s1/confirm');
        expect(jsonDecode(r.body), {'decision': 'corrected', 'paper_speed_mm_s': 50.0});
        return http.Response('{"scan_id": "s1", "status": "queued"}', 200);
      }));
      expect((await api.confirm('s1', 'corrected', paperSpeedMmS: 50))['status'], 'queued');
    });
  });

  group('upload queue', () {
    late Directory dir;
    setUp(() => dir = Directory.systemTemp.createTempSync('queue'));
    tearDown(() => dir.deleteSync(recursive: true));

    test('retries with a growing pause, survives a restart, deletes the sent photo', () async {
      var now = DateTime(2026, 1, 1, 12);
      var up = false;
      final client = MockClient.streaming((req, body) async {
        await body.drain<void>();
        if (!up) throw const SocketException('offline');
        return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s9"}')), 202);
      });
      final api = ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: client);
      final q = UploadQueue(dir, api, clock: () => now);
      await q.load();
      final item = await q.add(jpeg(10, 10, 200), const ScanHints(gainMmMv: 10));
      expect(await q.pump(), 0);
      expect(item.state, UploadState.queued);
      expect(q.retryIn(item), 5);
      now = now.add(const Duration(seconds: 3));
      expect(await q.pump(), 0); // not due yet: no attempt
      expect(item.attempts, 1);
      now = now.add(const Duration(seconds: 3));
      await q.pump();
      expect(q.retryIn(item), 10); // doubled

      final again = UploadQueue(dir, api, clock: () => now); // app restarted
      await again.load();
      expect(again.items.single.attempts, 2);
      up = true;
      now = now.add(const Duration(seconds: 11));
      expect(await again.pump(), 1);
      expect(again.items.single.state, UploadState.sent);
      expect(again.items.single.scanId, 's9');
      expect(File('${dir.path}/${item.localId}.jpg').existsSync(), isFalse);
    });

    test('a refusal is not retried', () async {
      final api = ScanApi(baseUrl: 'http://srv', apiKey: 'bad',
          client: MockClient.streaming((req, body) async {
        await body.drain<void>();
        return http.StreamedResponse(Stream.value(utf8.encode('{"detail": "invalid key"}')), 401);
      }));
      final q = UploadQueue(dir, api);
      await q.load();
      final item = await q.add(jpeg(10, 10, 200), const ScanHints());
      await q.pump();
      expect(item.state, UploadState.failed);
      expect(item.error, 'invalid key');
    });
  });
}
