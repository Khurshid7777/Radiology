// TZ addendum S2 (§18.4, §18.8): a study on several sheets and the offline queue.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/security/local_vault.dart';
import 'package:ecg_scanner/state/app_state.dart';
import 'package:ecg_scanner/state/study_status.dart';
import 'package:ecg_scanner/ui/sheets_screen.dart';
import 'package:ecg_scanner/upload/upload_queue.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;

/// A sheet told from the others by its size (side `w` px).
Uint8List sheet(int w) => Uint8List.fromList(img.encodeJpg(img.Image(width: w, height: w)..clear(img.ColorRgb8(200, 200, 200))));

/// A server that can be cut off: every request fails while `down`.
class Net {
  bool down = true;
  final List<List<String>> uploads = [];         // per accepted upload: the field names of its files
  final List<Map<String, String>> fields = [];
  final List<List<int>> lengths = [];

  late final MockClient client = MockClient.streaming((req, body) async {
    final bytes = await body.toBytes();
    if (down) throw const SocketException('network is down');
    final r = req as http.MultipartRequest;
    uploads.add([for (final f in r.files) f.field]);
    fields.add(Map.of(r.fields));
    lengths.add([for (final f in r.files) f.length]);
    expect(bytes, isNotEmpty);
    return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s${uploads.length}"}')), 202);
  });
}

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('s2'));
  tearDown(() => dir.deleteSync(recursive: true));

  test('no study is lost when the network drops: drafts, restarts, retries (§18.8)', () async {
    var now = DateTime(2026, 1, 1, 12);
    final net = Net();
    final keys = MemoryKeyStore();
    UploadQueue open() => UploadQueue(dir, ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: net.client),
        clock: () => now, vault: LocalVault(keys));

    // Three sheets taken offline; the order changed; the app is killed before sending.
    var q = open();
    await q.load();
    for (final w in [8, 64, 256]) {
      await q.addSheet(sheet(w), const ScanHints(headerMasked: true));
    }
    await q.moveSheet(q.draft!, 2, 0);                 // the last one goes first
    expect(await q.pump(), 0);                         // a draft is never sent on its own
    expect(net.uploads, isEmpty);
    for (final f in dir.listSync().whereType<File>().where((f) => f.path.endsWith('.bin'))) {
      expect(f.readAsBytesSync().length, greaterThan(0));
      expect(f.readAsBytesSync().sublist(0, 3), isNot([0xff, 0xd8, 0xff]));   // sealed, not a JPEG
    }

    q = open();                                        // restarted: the draft is there, in its order
    await q.load();
    final d = q.draft!;
    expect(d.partIds, hasLength(3));
    expect(studyStatus(d, null), StudyStatus.draft);
    await q.removeSheet(d, 1);                         // the 8-px sheet removed
    await q.send(d, const ScanHints(paperSpeedMmS: 50));
    expect(studyStatus(d, null), StudyStatus.queued);

    // Offline: tried, kept, the next try later; another restart in between.
    expect(await q.pump(), 0);
    expect(d.state, UploadState.queued);
    expect(d.attempts, 1);
    q = open();
    await q.load();
    now = now.add(const Duration(minutes: 1));
    net.down = false;
    expect(await q.pump(), 1);
    final sent = q.items.single;
    expect(sent.state, UploadState.sent);
    expect(net.uploads.single, ['file', 'parts']);     // the sheets in the doctor's order
    expect(net.lengths.single, [sheet(256).length, sheet(64).length]);
    expect(net.fields.single['pages'], 'true');
    expect(net.fields.single['paper_speed_mm_s'], '50.0');
    expect(dir.listSync().whereType<File>().where((f) => f.path.endsWith('.bin')), isEmpty);   // sent: deleted

    expect(studyStatus(sent, null), StudyStatus.processing);
    expect(studyStatus(sent, {'status': 'processing'}), StudyStatus.processing);
    expect(studyStatus(sent, {'status': 'needs_confirmation'}), StudyStatus.ready);
    expect(studyStatus(sent, {'status': 'rejected'}), StudyStatus.retake);
  });

  test('a refused upload waits for the doctor and is sent again on request', () async {
    var key = 'bad';
    final client = MockClient.streaming((req, body) async {
      await body.drain<void>();
      if (req.headers['X-API-Key'] == 'bad') {
        return http.StreamedResponse(Stream.value(utf8.encode('{"detail": "invalid key"}')), 401);
      }
      return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s1"}')), 202);
    });
    final q = UploadQueue(dir, ScanApi(baseUrl: 'http://srv', apiKey: key, client: client));
    await q.load();
    final item = await q.add(sheet(100), const ScanHints());
    await q.pump();
    expect(studyStatus(item, null), StudyStatus.failed);
    expect(dir.listSync().whereType<File>().where((f) => f.path.endsWith('.jpg')), hasLength(1));   // still on the phone
    key = 'good';
    q.api = ScanApi(baseUrl: 'http://srv', apiKey: key, client: client);
    await q.retry(item);
    expect(await q.pump(), 1);
    expect(item.state, UploadState.sent);
  });

  testWidgets('sheets screen: the draft in order, a sheet removed, the study sent', (tester) async {
    final net = Net()..down = false;
    final app = await tester.runAsync(() async {
      final a = await AppState.open(dir, api: ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: net.client));
      for (final w in [16, 32]) {
        await a.queue.addSheet(sheet(w), const ScanHints());
      }
      return a;
    }) as AppState;
    await tester.pumpWidget(MaterialApp(home: Builder(builder: (context) => SheetsScreen(app: app))));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    expect(find.text('Лист 1'), findsOneWidget);
    expect(find.text('Лист 2'), findsOneWidget);
    // The screen's file work (the sealed sheets on disk) runs in real time: the taps too.
    Future<void> tapReal(Finder f) async {
      await tester.runAsync(() async {
        await tester.tap(f);
        await Future<void>.delayed(const Duration(milliseconds: 150));
      });
      await tester.pump();
    }

    await tapReal(find.byKey(const Key('remove_0')));
    expect(find.text('Лист 2'), findsNothing);
    expect(app.queue.draft!.partIds, hasLength(1));
    await tester.tap(find.text('50 мм/с'));
    await tester.pump();
    expect(tester.widget<ButtonStyleButton>(find.byKey(const Key('send_study'))).onPressed, isNotNull);
    await tapReal(find.byKey(const Key('send_study')));
    expect(app.queue.draft, isNull);
    expect(app.queue.items.single.hints.paperSpeedMmS, 50);
    expect(app.queue.items.single.hints.pages, isFalse);       // one sheet left: not a multi-page study
  });

  test('no space left: the photo stays in memory, is sent from there, and written later (§18.8)', () async {
    var full = true;
    var up = false;
    final client = MockClient.streaming((req, body) async {
      await body.drain<void>();
      if (!up) throw const SocketException('offline');
      return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s1"}')), 202);
    });
    Future<void> writer(File f, List<int> b) async {
      if (full) throw FileSystemException('No space left on device', f.path);
      await f.writeAsBytes(b, flush: true);
    }

    var now = DateTime(2026, 1, 1);
    final q = UploadQueue(dir, ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: client),
        clock: () => now, writeFile: writer);
    await q.load();
    final item = await q.add(sheet(32), const ScanHints());     // the disk refuses: kept in memory
    expect(q.diskFull, isTrue);
    expect(q.items.single, same(item));
    await q.pump();                                              // offline too: still kept
    expect(item.state, UploadState.queued);
    full = false;
    now = now.add(const Duration(minutes: 1));
    await q.pump();                                              // space again: written at the next pump
    expect(q.diskFull, isFalse);
    expect(File('${dir.path}/index.json').existsSync(), isTrue);
    expect(File('${dir.path}/${item.localId}.jpg').existsSync(), isTrue);
    up = true;
    now = now.add(const Duration(minutes: 1));
    expect(await q.pump(), 1);
    expect(item.state, UploadState.sent);
  });
}
