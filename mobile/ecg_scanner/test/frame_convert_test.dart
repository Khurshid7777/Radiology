import 'dart:typed_data';

import 'package:ecg_scanner/camera/frame_convert.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('YUV420: neutral grey stays grey, red is red; the step reduces by block means', () {
    const w = 8, h = 4;
    final y = Uint8List(w * h)..fillRange(0, w * h, 128);
    y[0] = 0; // one dark pixel: block mean, not a skip
    final u = Uint8List((w ~/ 2) * (h ~/ 2))..fillRange(0, 8, 128);
    final v = Uint8List((w ~/ 2) * (h ~/ 2))..fillRange(0, 8, 128);
    final grey = yuv420ToRgb(width: w, height: h, y: y, yRowStride: w, u: u, v: v, uvRowStride: w ~/ 2,
        uvPixelStride: 1, targetWidth: 4);
    expect(grey.step, 2);
    expect([grey.width, grey.height], [4, 2]);
    expect(grey.rgb.sublist(0, 3), [96, 96, 96]); // (0 + 3 * 128) / 4
    expect(grey.rgb.sublist(3, 6), [128, 128, 128]);
    v.fillRange(0, 8, 255);
    final red = yuv420ToRgb(width: w, height: h, y: y, yRowStride: w, u: u, v: v, uvRowStride: w ~/ 2,
        uvPixelStride: 1, targetWidth: 8);
    expect(red.rgb[3], greaterThan(red.rgb[4] + 100));
  });

  test('BGRA: channel order and step', () {
    final bytes = Uint8List.fromList([10, 20, 30, 255, 10, 20, 30, 255]);
    final f = bgraToRgb(width: 2, height: 1, bytes: bytes, rowStride: 8, targetWidth: 2);
    expect(f.rgb, [30, 20, 10, 30, 20, 10]);
  });
}
