// Parity with the server's zone mapping (S3): the reference vectors are the Python ones in
// tests/scanner/fixtures/mask_zones (src/ecgscan/masking.py).
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:ecg_scanner/masking/zone_mask.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

final fixtures = Directory('../../tests/scanner/fixtures/mask_zones');

List<List<double>> _quad(dynamic q) => [for (final p in q as List) [(p[0] as num).toDouble(), (p[1] as num).toDouble()]];

void main() {
  final ref = jsonDecode(File('${fixtures.path}/vectors.json').readAsStringSync()) as Map<String, dynamic>;
  final tol = (ref['tolerance_px'] as num).toDouble();
  final cases = ref['cases'] as List;

  test('every reference zone maps onto the same photo polygon', () {
    expect(cases.length, greaterThanOrEqualTo(48));
    for (final c in cases) {
      final got = zonePolygon([for (final v in c['zone'] as List) (v as num).toDouble()], _quad(c['corners']),
          c['rotate_deg'] as int, margin: (c['margin'] as num).toDouble());
      final want = _quad(c['polygon']);
      for (var i = 0; i < 4; i++) {
        expect((got[i][0] - want[i][0]).abs(), lessThanOrEqualTo(tol), reason: '${c['name']} ${c['rotate_deg']}');
        expect((got[i][1] - want[i][1]).abs(), lessThanOrEqualTo(tol), reason: '${c['name']} ${c['rotate_deg']}');
      }
    }
  });

  test('a zone is painted white and nothing else is', () {
    final photo = img.Image(width: 400, height: 300);
    img.fill(photo, color: img.ColorRgb8(0, 0, 0));
    final jpeg = Uint8List.fromList(img.encodeJpg(photo, quality: 100));
    const corners = [
      [0.0, 0.0],
      [400.0, 0.0],
      [400.0, 300.0],
      [0.0, 300.0],
    ];
    final out = img.decodeJpg(maskZones(jpeg, [
      [0.0, 0.0, 0.5, 0.1]
    ], corners, 0))!;
    expect(out.getPixel(20, 10).r, greaterThan(240));      // inside the zone (x < 200, y < 30)
    expect(out.getPixel(300, 10).r, lessThan(15));          // right of it
    expect(out.getPixel(20, 200).r, lessThan(15));          // below it
  });

  test("a quarter turn masks the photo's left strip for the sheet's top strip", () {
    final photo = img.Image(width: 300, height: 400);
    img.fill(photo, color: img.ColorRgb8(0, 0, 0));
    final jpeg = Uint8List.fromList(img.encodeJpg(photo, quality: 100));
    const corners = [
      [0.0, 0.0],
      [300.0, 0.0],
      [300.0, 400.0],
      [0.0, 400.0],
    ];
    final out = img.decodeJpg(maskZones(jpeg, [
      [0.0, 0.0, 1.0, 0.1]
    ], corners, 90))!;
    expect(out.getPixel(10, 200).r, greaterThan(240));
    expect(out.getPixel(150, 10).r, lessThan(15));
  });
}
