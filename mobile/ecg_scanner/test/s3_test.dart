// TZ addendum S3 (§18.6): personal-data zones from the recorder's profile masked before upload.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:ecg_scanner/state/app_state.dart';
import 'package:ecg_scanner/state/settings.dart';
import 'package:ecg_scanner/ui/crop_screen.dart';
import 'package:ecg_scanner/ui/mask_review_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;

const corners = [
  [0.0, 0.0],
  [400.0, 0.0],
  [400.0, 300.0],
  [0.0, 300.0],
];
const profile = {
  'id': 'ek12',
  'manufacturer': 'Test',
  'model': 'EK-12',
  'layouts': ['3x4+rhythm'],
  'phi_zones': [
    [0.0, 0.0, 0.5, 0.1]
  ],
  'phi_zone_margin': 0.0,
};

Uint8List black() => Uint8List.fromList(img.encodeJpg(img.Image(width: 400, height: 300), quality: 100));

/// The first JPEG in a multipart body.
img.Image uploadedPhoto(List<int> body) {
  for (var i = 0; i + 1 < body.length; i++) {
    if (body[i] == 0xff && body[i + 1] == 0xd8) {
      for (var j = body.length - 2; j > i; j--) {
        if (body[j] == 0xff && body[j + 1] == 0xd9) return img.decodeJpg(Uint8List.fromList(body.sublist(i, j + 2)))!;
      }
    }
  }
  throw StateError('no JPEG in the upload');
}

class Server {
  final List<Map<String, String>> fields = [];
  final List<img.Image> photos = [];

  late final MockClient client = MockClient.streaming((req, body) async {
    final bytes = await body.toBytes();
    fields.add(Map.of((req as http.MultipartRequest).fields));
    photos.add(uploadedPhoto(bytes));
    return http.StreamedResponse(Stream.value(utf8.encode('{"scan_id": "s${fields.length}"}')), 202);
  });
}

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('s3'));
  tearDown(() => dir.deleteSync(recursive: true));

  Future<(AppState, Server)> open({bool device = true}) async {
    final srv = Server();
    final app = await AppState.open(dir, api: ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: srv.client));
    app.settings.maskHeader = true;
    app.settings.setDevice(device ? Map<String, dynamic>.from(profile) : null);
    return (app, srv);
  }

  Map<String, dynamic> capture({int rotate = 0}) => {
        'capture_mode': 'custom_camera',
        'corners': {'auto': null, 'final': corners, 'edited_by_user': false},
        'rotate_deg': rotate,
      };

  test("the profile's zones are painted white, the rest of the header is left (S3)", () async {
    final (app, srv) = await open();
    await app.addPhoto(black(), band: 0.0, capture: capture());
    await app.queue.pump();
    final photo = srv.photos.single;
    expect(photo.getPixel(20, 10).r, greaterThan(240));        // name / ID zone
    expect(photo.getPixel(300, 10).r, lessThan(15));            // speed, gain, measurements: kept
    expect(photo.getPixel(20, 200).r, lessThan(15));
    final f = srv.fields.single;
    expect(f['device_profile'], 'ek12');
    expect(f['header_masked'], 'true');
    expect((jsonDecode(f['capture']!) as Map)['mask_method'], 'profile_zones');
  });

  test("the doctor's band on top of the zones is the manual correction", () async {
    final (app, srv) = await open();
    await app.addPhoto(black(), band: 0.2, capture: capture());
    await app.queue.pump();
    expect(srv.photos.single.getPixel(300, 10).r, greaterThan(240));
    expect((jsonDecode(srv.fields.single['capture']!) as Map)['mask_method'], 'manual');
  });

  test('without a profile the top band is masked as before', () async {
    final (app, srv) = await open(device: false);
    await app.addPhoto(black(), capture: capture());
    await app.queue.pump();
    expect(srv.photos.single.getPixel(300, 10).r, greaterThan(240));   // the whole band
    expect(srv.fields.single.containsKey('device_profile'), isFalse);
    expect((jsonDecode(srv.fields.single['capture']!) as Map)['mask_method'], 'template_zone');
  });

  test('the chosen recorder and its zones survive a restart (offline masking)', () async {
    final s = Settings()..setDevice(Map<String, dynamic>.from(profile));
    final back = Settings.fromJson(jsonDecode(jsonEncode(s.toJson())) as Map<String, dynamic>);
    expect(back.deviceProfileId, 'ek12');
    expect(back.deviceProfileName, 'Test EK-12');
    expect(back.deviceZones, [
      [0.0, 0.0, 0.5, 0.1]
    ]);
    expect(back.hasZones, isTrue);
  });

  testWidgets('the review screen shows the zones and starts with no band', (tester) async {
    final (app, _) = (await tester.runAsync(() => open()))!;
    final crop = CropResult(auto: null, corners: corners, edited: false, rotateDeg: 0, width: 400, height: 300);
    await tester.pumpWidget(MaterialApp(home: MaskReviewScreen(app: app, photo: black(), crop: crop)));
    await tester.pump();
    expect(find.byKey(const Key('zones')), findsOneWidget);
    expect(tester.widget<Slider>(find.byKey(const Key('band'))).value, 0.0);
  });

  testWidgets('without a profile the review screen is the band', (tester) async {
    final (app, _) = (await tester.runAsync(() => open(device: false)))!;
    await tester.pumpWidget(MaterialApp(home: MaskReviewScreen(app: app, photo: black())));
    await tester.pump();
    expect(find.byKey(const Key('zones')), findsNothing);
    expect(tester.widget<Slider>(find.byKey(const Key('band'))).value, app.settings.headerBand);
  });
}
