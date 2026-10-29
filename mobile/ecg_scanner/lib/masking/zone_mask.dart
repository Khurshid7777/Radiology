// Personal-data zones of a recorder's sheet masked before upload (S3, TZ 18.6; owner's
// decision 4, SD-070): a zone is [x0, y0, x1, y1] in fractions of the upright sheet, from the
// device profile's `phi_zones`. The sheet is the quadrilateral the doctor confirmed (TL, TR, BR,
// BL in photo pixels) turned clockwise by the crop's quarter turns. A zone grows by `margin`
// (fractions of the sheet) before mapping. Twin of src/ecgscan/masking.py; the reference vectors
// in tests/scanner/fixtures/mask_zones keep the two in agreement.
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// The photo corners of the upright sheet's TL, TR, BR, BL.
List<List<double>> sheetCorners(List<List<double>> corners, int rotateDeg) {
  final k = (rotateDeg ~/ 90) % 4;
  return [for (var i = 0; i < 4; i++) [corners[(i - k) % 4][0], corners[(i - k) % 4][1]]];
}

/// 3x3 map (row-major, 9 values) of the unit square onto `quad`.
List<double> homography(List<List<double>> quad) {
  const src = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
  ];
  final a = <List<double>>[];
  for (var i = 0; i < 4; i++) {
    final x = src[i][0], y = src[i][1], u = quad[i][0], v = quad[i][1];
    a.add([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    a.add([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  // Gaussian elimination with partial pivoting on the 8x9 augmented matrix.
  for (var c = 0; c < 8; c++) {
    var p = c;
    for (var r = c + 1; r < 8; r++) {
      if (a[r][c].abs() > a[p][c].abs()) p = r;
    }
    final t = a[c];
    a[c] = a[p];
    a[p] = t;
    for (var r = 0; r < 8; r++) {
      if (r == c) continue;
      final f = a[r][c] / a[c][c];
      for (var k = c; k < 9; k++) {
        a[r][k] -= f * a[c][k];
      }
    }
  }
  return [for (var i = 0; i < 8; i++) a[i][8] / a[i][i], 1.0];
}

double _round2(double v) => (v * 100).roundToDouble() / 100;

/// The zone's TL, TR, BR, BL in photo pixels.
List<List<double>> zonePolygon(List<double> zone, List<List<double>> corners, int rotateDeg, {double margin = 0}) {
  double clamp(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);
  final x0 = clamp(zone[0] - margin), y0 = clamp(zone[1] - margin);
  final x1 = clamp(zone[2] + margin), y1 = clamp(zone[3] + margin);
  final h = homography(sheetCorners(corners, rotateDeg));
  List<double> map(double x, double y) {
    final w = h[6] * x + h[7] * y + h[8];
    return [_round2((h[0] * x + h[1] * y + h[2]) / w), _round2((h[3] * x + h[4] * y + h[5]) / w)];
  }

  return [map(x0, y0), map(x1, y0), map(x1, y1), map(x0, y1)];
}

/// The photo with every zone painted white (and, if `extraBand` > 0, its top band too: the
/// doctor's correction), re-encoded as JPEG. EXIF is not carried over.
Uint8List maskZones(Uint8List jpeg, List<List<double>> zones, List<List<double>> corners, int rotateDeg,
    {double margin = 0, double extraBand = 0, int quality = 95}) {
  final photo = img.decodeImage(jpeg);
  if (photo == null) throw ArgumentError('not a decodable image');
  final upright = img.bakeOrientation(photo);
  final white = img.ColorRgb8(255, 255, 255);
  for (final z in zones) {
    final poly = zonePolygon(z, corners, rotateDeg, margin: margin);
    img.fillPolygon(upright, vertices: [for (final p in poly) img.Point(p[0], p[1])], color: white);
  }
  if (extraBand > 0) {
    final band = (upright.height * extraBand).round();
    img.fillRect(upright, x1: 0, y1: 0, x2: upright.width - 1, y2: band - 1, color: white);
  }
  upright.exif = img.ExifData();
  return Uint8List.fromList(img.encodeJpg(upright, quality: quality));
}
