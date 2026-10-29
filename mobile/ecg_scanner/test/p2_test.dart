// P2 (TZ 18.11, SD-097): the app asks which extension features are on; absent means off.
import 'dart:convert';

import 'package:ecg_scanner/api/scan_api.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('features are read from the server', () async {
    final api = ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: MockClient((req) async {
      expect(req.url.path, '/scans/config/features');
      return http.Response(jsonEncode({'features': {'consult': false, 'archive': false, 'qr': true, 'signature': false}}), 200);
    }));
    expect(await api.features(), {'consult': false, 'archive': false, 'qr': true, 'signature': false});
  });

  test('a server without the endpoint means every feature off', () async {
    final api = ScanApi(baseUrl: 'http://srv', apiKey: 'k', client: MockClient((req) async => http.Response('{}', 404)));
    expect(await api.features(), isEmpty);
  });
}
