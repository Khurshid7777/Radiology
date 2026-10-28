import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/state/app_state.dart';
import 'package:ecg_scanner/ui/confirm_screen.dart';
import 'package:ecg_scanner/ui/home_screen.dart';
import 'package:ecg_scanner/ui/mask_review_screen.dart';
import 'package:ecg_scanner/ui/settings_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;

Map<String, dynamic> scanBody(String status, {double? speed = 25, List<String> warnings = const []}) => {
      'scan_id': 's1',
      'status': status,
      'messages': {
        'ru': {'status': 'Статус $status', 'reasons': <String>[], 'warnings': warnings},
        'uz': {'status': 'Holat $status', 'reasons': <String>[], 'warnings': warnings},
      },
      'result': {
        'status': status,
        'calibration': {'paper_speed_mm_s': speed, 'gain_mm_mv': 10},
      },
    };

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('app'));
  tearDown(() => dir.deleteSync(recursive: true));

  Future<(AppState, List<http.Request>)> open(Map<String, dynamic> Function() current) async {
    final seen = <http.Request>[];
    final client = MockClient((req) async {
      seen.add(req);
      if (req.url.path.endsWith('/artifacts/trace_overlay_preview')) return http.Response('nope', 404);
      return http.Response.bytes(utf8.encode(jsonEncode(current())), 200,
          headers: {'content-type': 'application/json; charset=utf-8'});
    });
    final app = await AppState.open(dir, api: ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: client));
    return (app, seen);
  }

  testWidgets('settings: header masking is on by default', (tester) async {
    final (app, _) = await tester.runAsync(() => open(() => scanBody('ok'))) as (AppState, List<http.Request>);
    await tester.pumpWidget(MaterialApp(home: SettingsScreen(app: app)));
    final sw = tester.widget<SwitchListTile>(find.byKey(const Key('mask_header')));
    expect(sw.value, isTrue);
    expect(find.text('Скрывать ФИО и ID в шапке'), findsOneWidget);
  });

  testWidgets('home: empty history invites a capture', (tester) async {
    final (app, _) = await tester.runAsync(() => open(() => scanBody('ok'))) as (AppState, List<http.Request>);
    await tester.pumpWidget(MaterialApp(home: HomeScreen(app: app)));
    expect(find.textContaining('Снимков пока нет'), findsOneWidget);
    expect(find.text('Снять ЭКГ'), findsOneWidget);
  });

  testWidgets('confirm: unknown speed cannot be accepted; a correction is sent', (tester) async {
    var body = scanBody('needs_confirmation', speed: null, warnings: ['Скорость не установлена']);
    final (app, seen) = await tester.runAsync(() => open(() => body)) as (AppState, List<http.Request>);
    await tester.pumpWidget(MaterialApp(home: ConfirmScreen(app: app, scanId: 's1')));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    expect(find.text('Статус needs_confirmation'), findsOneWidget);
    expect(find.text('• Скорость не установлена'), findsOneWidget);
    expect(tester.widget<FilledButton>(find.byKey(const Key('accept'))).onPressed, isNull);
    await tester.tap(find.text('25 мм/с'));
    await tester.pumpAndSettle();
    expect(tester.widget<FilledButton>(find.byKey(const Key('accept'))).onPressed, isNull); // changed: correct, not accept
    body = scanBody('ok');
    await tester.tap(find.byKey(const Key('correct')));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    final post = seen.firstWhere((r) => r.url.path == '/scans/s1/confirm');
    expect(jsonDecode(post.body), {'decision': 'corrected', 'paper_speed_mm_s': 25.0, 'gain_mm_mv': 10.0});
    expect(find.text('Статус ok'), findsOneWidget);
    expect(find.byKey(const Key('analysis')), findsOneWidget);
  });

  testWidgets('confirm: an unknown layout is asked for and sent as a correction (S2)', (tester) async {
    var body = scanBody('rejected');
    (body['result'] as Map<String, dynamic>)
      ..['reject_reasons'] = ['LAYOUT_UNKNOWN']
      ..['layout'] = {'name': null, 'confidence': 0.0};
    final (app, seen) = await tester.runAsync(() => open(() => body)) as (AppState, List<http.Request>);
    await tester.pumpWidget(MaterialApp(home: ConfirmScreen(app: app, scanId: 's1')));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('layout_ask')), findsOneWidget);
    expect(find.byKey(const Key('accept')), findsNothing);                    // a rejected scan is not accepted
    expect(tester.widget<OutlinedButton>(find.byKey(const Key('correct_layout'))).onPressed, isNull);
    await tester.tap(find.byKey(const Key('layout_6x2')));
    await tester.pumpAndSettle();
    body = scanBody('ok');
    await tester.tap(find.byKey(const Key('correct_layout')));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    final post = seen.firstWhere((r) => r.url.path == '/scans/s1/confirm');
    expect(jsonDecode(post.body), {'decision': 'corrected', 'layout': '6x2'});
    expect(find.text('Статус ok'), findsOneWidget);
  });

  testWidgets('mask review: the band can be moved, send returns it', (tester) async {
    final (app, _) = await tester.runAsync(() => open(() => scanBody('ok'))) as (AppState, List<http.Request>);
    double? got = -1;
    final photo = Uint8List.fromList(img.encodePng(img.Image(width: 40, height: 30)));
    await tester.pumpWidget(MaterialApp(home: Builder(builder: (context) => TextButton(
        onPressed: () async => got = await Navigator.push<double>(
            context, MaterialPageRoute(builder: (_) => MaskReviewScreen(app: app, photo: photo))),
        child: const Text('go')))));
    await tester.tap(find.text('go'));
    await tester.pumpAndSettle();
    expect(tester.widget<Slider>(find.byKey(const Key('band'))).value, 0.14);
    await tester.drag(find.byKey(const Key('band')), const Offset(200, 0));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('send')));
    await tester.pumpAndSettle();
    expect(got, greaterThan(0.14));
  });

  testWidgets('confirm: tablet width puts the panel beside the picture', (tester) async {
    tester.view.physicalSize = const Size(1600, 1000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);
    final (app, _) = await tester.runAsync(() => open(() => scanBody('ok'))) as (AppState, List<http.Request>);
    await tester.pumpWidget(MaterialApp(home: ConfirmScreen(app: app, scanId: 's1')));
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 50)));
    await tester.pumpAndSettle();
    expect(find.byType(Row), findsWidgets);
    expect(tester.getSize(find.byType(ListView)).width, 380);
  });
}
