// The sheet found by the frame checks, drawn over the live preview (TZ addendum §18.2).
//
// The checks run on the camera's frames, which come in the sensor's orientation; the
// preview on a portrait screen is that image turned by `sensorOrientation` degrees
// clockwise. Corners are mapped to 0..1 of the frame, then turned the same way.
import 'package:flutter/material.dart';

/// Normalised frame point (0..1) to the normalised point on the upright preview.
Offset turnNormalized(Offset p, int sensorOrientation) {
  switch (((sensorOrientation % 360) + 360) % 360) {
    case 90:
      return Offset(1 - p.dy, p.dx);
    case 180:
      return Offset(1 - p.dx, 1 - p.dy);
    case 270:
      return Offset(p.dy, 1 - p.dx);
    default:
      return p;
  }
}

/// The sheet corners (captured-frame px) as normalised preview points, TL TR BR BL order kept.
List<Offset>? previewQuad(List<List<double>>? corners, double frameWidth, double frameHeight, int sensorOrientation) {
  if (corners == null || frameWidth <= 0 || frameHeight <= 0) return null;
  return [for (final c in corners) turnNormalized(Offset(c[0] / frameWidth, c[1] / frameHeight), sensorOrientation)];
}

class SheetOverlayPainter extends CustomPainter {
  SheetOverlayPainter(this.quad, {required this.ready});

  final List<Offset>? quad;
  final bool ready;

  @override
  void paint(Canvas canvas, Size size) {
    final q = quad;
    if (q == null) return;
    final path = Path()..moveTo(q[0].dx * size.width, q[0].dy * size.height);
    for (final p in q.skip(1)) {
      path.lineTo(p.dx * size.width, p.dy * size.height);
    }
    path.close();
    canvas.drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = ready ? Colors.greenAccent : Colors.amberAccent);
  }

  @override
  bool shouldRepaint(SheetOverlayPainter old) => old.quad != quad || old.ready != ready;
}
