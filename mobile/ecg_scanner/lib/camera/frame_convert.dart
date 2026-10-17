// Camera frames -> RGB for the frame checks (TZ §6.1).
//
// The camera runs at full resolution (a whole page needs >= 8 px/mm in the photo,
// TZ §6.1; at 1920 px a 280-mm page is ~6.9 px/mm), so preview frames are reduced
// here by an integer step with block means of luma (the grid's thin lines survive a
// mean, not a skip) and the chroma sample of each block. The step is the
// `scale_to_capture` of the checks.
import 'dart:math' as math;
import 'dart:typed_data';

class RgbFrame {
  RgbFrame(this.rgb, this.width, this.height, this.step);

  final Uint8List rgb;
  final int width;
  final int height;

  /// Frame pixels per converted pixel (= captured photo width / converted width).
  final int step;
}

int stepFor(int width, int targetWidth) => math.max(1, (width / targetWidth).ceil());

int _clamp(double v) => v < 0 ? 0 : (v > 255 ? 255 : v.round());

/// Android YUV_420_888 (Y plane + U/V planes with their row and pixel strides).
RgbFrame yuv420ToRgb({
  required int width,
  required int height,
  required Uint8List y,
  required int yRowStride,
  required Uint8List u,
  required Uint8List v,
  required int uvRowStride,
  required int uvPixelStride,
  int targetWidth = 1920,
}) {
  final s = stepFor(width, targetWidth);
  final w = width ~/ s, h = height ~/ s;
  final out = Uint8List(w * h * 3);
  final n = (s * s).toDouble();
  for (var oy = 0; oy < h; oy++) {
    for (var ox = 0; ox < w; ox++) {
      var sum = 0;
      for (var dy = 0; dy < s; dy++) {
        final row = (oy * s + dy) * yRowStride + ox * s;
        for (var dx = 0; dx < s; dx++) {
          sum += y[row + dx];
        }
      }
      final yy = sum / n;
      final cx = (ox * s + s ~/ 2) >> 1, cy = (oy * s + s ~/ 2) >> 1;
      final ci = cy * uvRowStride + cx * uvPixelStride;
      final uu = u[ci] - 128.0, vv = v[ci] - 128.0;
      final o = (oy * w + ox) * 3;
      out[o] = _clamp(yy + 1.402 * vv);
      out[o + 1] = _clamp(yy - 0.344136 * uu - 0.714136 * vv);
      out[o + 2] = _clamp(yy + 1.772 * uu);
    }
  }
  return RgbFrame(out, w, h, s);
}

/// iOS BGRA8888 (one plane).
RgbFrame bgraToRgb({
  required int width,
  required int height,
  required Uint8List bytes,
  required int rowStride,
  int targetWidth = 1920,
}) {
  final s = stepFor(width, targetWidth);
  final w = width ~/ s, h = height ~/ s;
  final out = Uint8List(w * h * 3);
  final n = (s * s).toDouble();
  for (var oy = 0; oy < h; oy++) {
    for (var ox = 0; ox < w; ox++) {
      var b = 0, g = 0, r = 0;
      for (var dy = 0; dy < s; dy++) {
        var p = (oy * s + dy) * rowStride + ox * s * 4;
        for (var dx = 0; dx < s; dx++, p += 4) {
          b += bytes[p];
          g += bytes[p + 1];
          r += bytes[p + 2];
        }
      }
      final o = (oy * w + ox) * 3;
      out[o] = (r / n).round();
      out[o + 1] = (g / n).round();
      out[o + 2] = (b / n).round();
    }
  }
  return RgbFrame(out, w, h, s);
}
