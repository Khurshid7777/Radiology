// S1 of the TZ addendum (§18.2, §18.3, §18.9, §18.13): camera tilt, auto-capture on
// still corners, the sheet overlay, the crop screen, the capture record, the encrypted
// queue and the app lock.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/camera/sheet_overlay.dart';
import 'package:ecg_scanner/qc/auto_capture.dart';
import 'package:ecg_scanner/qc/camera_tilt.dart';
import 'package:ecg_scanner/qc/capture_qc.dart';
import 'package:ecg_scanner/security/app_lock.dart';
import 'package:ecg_scanner/security/local_vault.dart';
import 'package:ecg_scanner/state/app_state.dart';
import 'package:ecg_scanner/ui/camera_screen.dart';
import 'package:ecg_scanner/ui/crop_screen.dart';
import 'package:ecg_scanner/upload/upload_queue.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;

const square = [
  [100.0, 100.0],
  [900.0, 100.0],
  [900.0, 700.0],
  [100.0, 700.0],
];

FrameQc qc(List<String> hints, {List<List<double>>? corners = square}) => FrameQc(
    width: 1000, height: 800, factor: 1, brightness: 170, glare: 0, sharpness: 500, tiltDeg: 0.2, periodPx: 20,
    pxPerMm: 11, sheet: 0.6, hints: hints, corners: corners, shadow: 0.05, sheetWhole: true, sheetArea: 0.6);

List<List<double>> moved(double dx) => [for (final c in square) [c[0] + dx, c[1]]];

Uint8List jpeg(int w, int h, int grey) =>
    Uint8List.fromList(img.encodeJpg(img.Image(width: w, height: h)..clear(img.ColorRgb8(grey, grey, grey))));

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('camera tilt', () {
    test('flat is 0, 30 degrees off is 30, sign and axis do not matter', () {
      expect(cameraTiltDeg(0, 0, 9.81), closeTo(0, 1e-9));
      expect(cameraTiltDeg(0, 0, -9.81), closeTo(0, 1e-9));
      final r = 30 * math.pi / 180;
      expect(cameraTiltDeg(0, 9.81 * math.sin(r), 9.81 * math.cos(r)), closeTo(30, 1e-6));
      expect(cameraTiltDeg(9.81 * math.sin(r), 0, -9.81 * math.cos(r)), closeTo(30, 1e-6));
      final f = TiltFilter(alpha: 0.5);
      f.update(0, 0, 9.81);
      expect(f.update(0, 9.81, 0), lessThan(90)); // a single jolt is smoothed
    });
  });

  group('auto-capture on still corners (TZ §18.2)', () {
    final t0 = DateTime(2026, 10, 26, 12);
    DateTime at(int ms) => t0.add(Duration(milliseconds: ms));

    test('fires only after the corners held still for stable_ms', () {
      final a = AutoCapture(3, stableMs: 700, maxJitter: 0.02);
      expect(a.update(qc([]), now: at(0)).fire, isFalse);
      expect(a.update(qc([]), now: at(200)).fire, isFalse);
      expect(a.update(qc([]), now: at(400)).fire, isFalse); // 3 frames, but 0.4 s
      expect(a.update(qc([]), now: at(750)).fire, isTrue);
    });

    test('a corner that moves restarts the wait; a still one does not', () {
      final a = AutoCapture(2, stableMs: 700, maxJitter: 0.02);
      a.update(qc([]), now: at(0));
      a.update(qc([], corners: moved(5)), now: at(300)); // 5 px of a 1000-px diagonal: still
      a.update(qc([], corners: moved(80)), now: at(600)); // 8 %: moved
      expect(a.update(qc([], corners: moved(80)), now: at(1000)).fire, isFalse);
      expect(a.update(qc([], corners: moved(80)), now: at(1350)).fire, isTrue);
    });

    test('the camera tilt is one more check', () {
      final a = AutoCapture.fromConfig({
        'auto_capture': {'consecutive_frames': 1, 'stable_ms': 0, 'corner_jitter': 0.02},
        'thresholds': {'max_camera_tilt_deg': 10},
      });
      expect(a.update(qc([]), cameraTiltDeg: 14).hint, 'level');
      expect(a.update(qc(['glare']), cameraTiltDeg: 14).hint, 'glare'); // display order kept
      expect(a.update(qc([]), cameraTiltDeg: 4).fire, isTrue);
    });

    test('jitter is relative to the sheet', () {
      expect(cornerJitter(square, moved(10)), closeTo(10 / 1000, 1e-9));
      expect(cornerJitter(square, square), 0);
    });
  });

  group('sheet overlay', () {
    test('sensor frame corners turn onto the portrait preview', () {
      expect(turnNormalized(const Offset(0.1, 0.2), 90), const Offset(0.8, 0.1));
      expect(turnNormalized(const Offset(0.1, 0.2), 270), const Offset(0.2, 0.9));
      expect(turnNormalized(const Offset(0.1, 0.2), 0), const Offset(0.1, 0.2));
      final q = previewQuad(square, 1000, 800, 0)!;
      expect(q.first, const Offset(0.1, 0.125));
      expect(previewQuad(null, 1000, 800, 90), isNull);
    });
  });

  group('capture record', () {
    test('the frame checks travel with the photo as the contract names them', () {
      final r = qcRecord(qc([]), const []);
      expect(r['passed'], isTrue);
      expect(r.keys.toSet(), {'brightness', 'glare', 'sharpness', 'shadow', 'px_per_mm', 'tilt_deg', 'sheet_area',
          'passed', 'failed_checks'});
      expect(qcRecord(qc(['glare']), const ['glare'])['failed_checks'], ['glare']);
    });

    test('the record is a form field; hints survive the queue index', () {
      const h = ScanHints(capture: {'capture_mode': 'custom_camera', 'rotate_deg': 90}, device: 'Pixel 7');
      expect(jsonDecode(h.toFields()['capture']!)['rotate_deg'], 90);
      expect(h.toFields()['device'], 'Pixel 7');
      final back = ScanHints.fromJson(jsonDecode(jsonEncode(h.toJson())) as Map<String, dynamic>);
      expect(back.capture!['capture_mode'], 'custom_camera');
    });
  });

  group('encrypted queue (§18.9)', () {
    late Directory dir;
    setUp(() => dir = Directory.systemTemp.createTempSync('vault'));
    tearDown(() => dir.deleteSync(recursive: true));

    test('seal / open round trip; the key is made once and kept by the keystore', () async {
      final keys = MemoryKeyStore();
      final v = LocalVault(keys);
      final plain = jpeg(30, 20, 128);
      final sealed = await v.seal(plain);
      expect(sealed.length, plain.length + 28);
      expect(await LocalVault(keys).open(sealed), plain); // a new vault, the same stored key
      expect(keys.values.length, 1);
      sealed[20] ^= 1;
      await expectLater(v.open(sealed), throwsA(anything)); // tampering is detected
    });

    test('the waiting photo is never on disk in clear; the server gets the original bytes', () async {
      Uint8List? sent;
      final client = MockClient.streaming((req, body) async {
        final raw = await body.toBytes();
        final i = _indexOf(raw, [0xFF, 0xD8, 0xFF]);
        sent = Uint8List.fromList(raw.sublist(i, i + photo.length));
        return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s9", "status": "queued"}')), 202);
      });
      final q = UploadQueue(dir, ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: client),
          vault: LocalVault(MemoryKeyStore()));
      await q.load();
      final item = await q.add(photo, const ScanHints());
      final files = dir.listSync().whereType<File>().where((f) => !f.path.endsWith('.json')).toList();
      expect(files.single.path, endsWith('${item.localId}.bin'));
      expect(_indexOf(files.single.readAsBytesSync(), [0xFF, 0xD8, 0xFF]), -1);
      expect(await q.pump(), 1);
      expect(sent, photo);
      expect(dir.listSync().whereType<File>().where((f) => !f.path.endsWith('.json')), isEmpty);
    });
  });

  group('app lock (§18.9)', () {
    test('PIN hash: salted, repeatable, never the PIN; rules', () async {
      final salt = List<int>.filled(16, 7);
      final a = await hashPin('1234', salt, iterations: 1000);
      expect(a, await hashPin('1234', salt, iterations: 1000));
      expect(a, isNot(await hashPin('1234', List<int>.filled(16, 8), iterations: 1000)));
      expect(a.contains('1234'), isFalse);
      expect(sameHash(a, a), isTrue);
      expect(validPin('12'), isFalse);
      expect(validPin('12a4'), isFalse);
      expect(validPin('123456'), isTrue);
      final now = DateTime(2026);
      expect(shouldLock(now.subtract(const Duration(minutes: 6)), now, const Duration(minutes: 5)), isTrue);
      expect(shouldLock(now.subtract(const Duration(minutes: 2)), now, const Duration(minutes: 5)), isFalse);
      expect(shouldLock(null, now, Duration.zero), isFalse);
    });

    testWidgets('first start sets the PIN; the next start asks for it', (tester) async {
      final dir = Directory.systemTemp.createTempSync('lock');
      addTearDown(() => dir.deleteSync(recursive: true));
      final app = (await tester.runAsync(() => AppState.open(dir)))!;
      Future<void> enter(String pin) async {
        await tester.enterText(find.byKey(const Key('pin')), pin);
        await tester.tap(find.byKey(const Key('pin_ok')));
        // Hashing and the settings file are real async work: let it run between frames.
        for (var i = 0; i < 20; i++) {
          await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 30)));
          await tester.pump();
        }
        await tester.pumpAndSettle();
      }

      await tester.pumpWidget(MaterialApp(home: AppLock(app: app, iterations: 1000, child: const Text('inside'))));
      expect(find.text('Придумайте PIN-код (4–8 цифр)'), findsOneWidget);
      await enter('12');
      expect(find.text('PIN-код — от 4 до 8 цифр.'), findsOneWidget);
      await enter('2468');
      await enter('2468');
      expect(find.text('inside'), findsOneWidget);
      expect(app.settings.pinSet, isTrue);
      expect(File('${dir.path}/settings.json').readAsStringSync().contains('2468'), isFalse);

      await tester.pumpWidget(const SizedBox());
      await tester.pumpWidget(MaterialApp(home: AppLock(app: app, iterations: 1000, child: const Text('inside'))));
      expect(find.text('Введите PIN-код'), findsOneWidget);
      await enter('1111');
      expect(find.text('Неверный PIN-код.'), findsOneWidget);
      await enter('2468');
      expect(find.text('inside'), findsOneWidget);
    });
  });

  group('crop screen (§18.3)', () {
    Future<AppState> app(WidgetTester tester) async {
      final dir = Directory.systemTemp.createTempSync('crop');
      addTearDown(() => dir.deleteSync(recursive: true));
      return (await tester.runAsync(() => AppState.open(dir)))!;
    }

    testWidgets('corners start where the sheet was found; dragging, turning and done', (tester) async {
      final a = await app(tester);
      CropResult? got;
      var popped = false;
      await tester.pumpWidget(MaterialApp(
          home: Builder(
              builder: (context) => TextButton(
                  onPressed: () async {
                    got = await Navigator.push<CropResult>(
                        context,
                        MaterialPageRoute(
                            builder: (_) =>
                                CropScreen(app: a, photo: jpeg(400, 300, 200), sheet: PhotoSheet(400, 300, square2))));
                    popped = true;
                  },
                  child: const Text('go')))));
      await tester.tap(find.text('go'));
      await tester.pumpAndSettle();
      expect(find.text('Углы листа'), findsOneWidget);
      await tester.drag(find.byKey(const Key('corner_0')), const Offset(-20, -20));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('crop_rotate')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('crop_done')));
      await tester.pumpAndSettle();
      expect(popped, isTrue);
      expect(got!.edited, isTrue);
      expect(got!.rotateDeg, 90);
      expect(got!.auto, square2);
      expect(got!.corners[0][0], lessThan(square2[0][0]));
      expect(got!.corners[2], square2[2]);
      final rec = got!.toCaptureCorners();
      expect(rec['edited_by_user'], isTrue);
    });

    testWidgets('no sheet found: the corners start at the photo corners; retake gives null', (tester) async {
      final a = await app(tester);
      CropResult? got = CropResult(auto: null, corners: const [], edited: false, rotateDeg: 0);
      await tester.pumpWidget(MaterialApp(
          home: Builder(
              builder: (context) => TextButton(
                  onPressed: () async => got = await Navigator.push<CropResult>(
                      context,
                      MaterialPageRoute(
                          builder: (_) => CropScreen(app: a, photo: jpeg(400, 300, 200), sheet: PhotoSheet(400, 300, null)))),
                  child: const Text('go')))));
      await tester.tap(find.text('go'));
      await tester.pumpAndSettle();
      expect(wholePhoto(400, 300)[2], [400.0, 300.0]);
      await tester.tap(find.byKey(const Key('crop_retake')));
      await tester.pumpAndSettle();
      expect(got, isNull);
    });
  });
}

const square2 = [
  [40.0, 30.0],
  [360.0, 30.0],
  [360.0, 270.0],
  [40.0, 270.0],
];

final photo = jpeg(64, 48, 90);

int _indexOf(List<int> hay, List<int> needle) {
  for (var i = 0; i + needle.length <= hay.length; i++) {
    var ok = true;
    for (var j = 0; j < needle.length; j++) {
      if (hay[i + j] != needle[j]) {
        ok = false;
        break;
      }
    }
    if (ok) return i;
  }
  return -1;
}
