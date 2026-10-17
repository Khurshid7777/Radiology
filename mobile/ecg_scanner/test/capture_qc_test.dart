// Parity with the server's frame checks (SD-014): the reference frames and expected
// metrics are the Python ones in tests/scanner/fixtures/capture_qc.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:ecg_scanner/qc/capture_qc.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

final fixtures = Directory('../../tests/scanner/fixtures/capture_qc');

void main() {
  final config = jsonDecode(File('${fixtures.path}/config.json').readAsStringSync()) as Map<String, dynamic>;
  final cases = fixtures
      .listSync()
      .whereType<File>()
      .where((f) => f.path.endsWith('.json') && !f.path.endsWith('config.json'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));

  test('reference frames are there', () => expect(cases.length, greaterThanOrEqualTo(8)));

  for (final f in cases) {
    final name = f.uri.pathSegments.last.replaceAll('.json', '');
    test('frame $name', () {
      final ref = jsonDecode(f.readAsStringSync()) as Map<String, dynamic>;
      final m = ref['metrics'] as Map<String, dynamic>;
      final decoded = img.decodeJpg(File('${fixtures.path}/$name.jpg').readAsBytesSync())!;
      final rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
      final q = checkFrame(rgb, decoded.width, decoded.height, config,
          scaleToCapture: (ref['scale_to_capture'] as num).toDouble());
      expect(q.hints.toSet(), (ref['expected_hints'] as List).cast<String>().toSet(), reason: name);
      void rel(String key, double? got, double tol) {
        final want = (m[key] as num?)?.toDouble();
        if (want == null) {
          expect(got, isNull, reason: key);
        } else {
          expect(got, closeTo(want, tol * want.abs()), reason: key);
        }
      }

      void abs(String key, double? got, double tol) {
        final want = (m[key] as num?)?.toDouble();
        if (want == null) {
          expect(got, isNull, reason: key);
        } else {
          expect(got, closeTo(want, tol), reason: key);
        }
      }

      // JPEG: another decoder than the server's (see tests/scanner/test_capture_qc.py).
      rel('px_per_mm', q.pxPerMm, 0.02);
      rel('sharpness', q.sharpness, 0.10);
      rel('brightness', q.brightness, 0.01);
      abs('tilt_deg', q.tiltDeg, 0.5);
      abs('sheet', q.sheet, 0.25);
      abs('glare', q.glare, 0.005);
    });
  }

  test('lossless frame: the same numbers as Python', () {
    final ref = jsonDecode(File('${fixtures.path}/exact/parity.json').readAsStringSync()) as Map<String, dynamic>;
    final m = ref['metrics'] as Map<String, dynamic>;
    final decoded = img.decodePng(File('${fixtures.path}/exact/parity.png').readAsBytesSync())!;
    final q = checkFrame(decoded.getBytes(order: img.ChannelOrder.rgb), decoded.width, decoded.height, config,
        scaleToCapture: (ref['scale_to_capture'] as num).toDouble());
    expect(q.hints, (m['hints'] as List).cast<String>());
    final got = {
      'brightness': q.brightness, 'glare': q.glare, 'sharpness': q.sharpness, 'tilt_deg': q.tiltDeg,
      'period_px': q.periodPx, 'px_per_mm': q.pxPerMm, 'sheet': q.sheet,
    };
    const decimals = {
      'brightness': 3, 'glare': 5, 'sharpness': 3, 'tilt_deg': 3, 'period_px': 3, 'px_per_mm': 3, 'sheet': 4,
    };
    decimals.forEach((key, dec) {
      expect(got[key], closeTo((m[key] as num).toDouble(), 0.6 * math.pow(10, -dec)), reason: key);
    });
  });
}
