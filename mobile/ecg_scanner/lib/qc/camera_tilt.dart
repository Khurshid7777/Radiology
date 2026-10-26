// Camera tilt from the accelerometer (TZ addendum §18.2, SD-072): a sheet lies on a desk,
// so the phone is parallel to it when gravity runs along the camera axis (device z).
// The frame checks cannot see this from the picture: perspective looks like a page
// shape, not like a tilt.
import 'dart:math' as math;

/// Angle (degrees) between the device z axis and gravity; 0 = screen up, lying flat.
double cameraTiltDeg(double ax, double ay, double az) {
  final n = math.sqrt(ax * ax + ay * ay + az * az);
  if (n < 1e-6) return 0.0;
  return math.acos(math.min(1.0, az.abs() / n)) * 180.0 / math.pi;
}

/// Exponential smoothing of the accelerometer: a hand's tremor is not a tilt.
class TiltFilter {
  TiltFilter({this.alpha = 0.2});

  final double alpha;
  double? _x, _y, _z;

  double update(double ax, double ay, double az) {
    _x = _x == null ? ax : _x! + alpha * (ax - _x!);
    _y = _y == null ? ay : _y! + alpha * (ay - _y!);
    _z = _z == null ? az : _z! + alpha * (az - _z!);
    return cameraTiltDeg(_x!, _y!, _z!);
  }

  double? get value => _x == null ? null : cameraTiltDeg(_x!, _y!, _z!);
}
