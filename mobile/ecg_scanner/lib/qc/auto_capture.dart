// Auto-capture (TZ §6.1, addendum §18.2): the shot is taken when every check passes on
// several consecutive frames and the sheet's corners have stayed put for `stableMs`
// (~0.7 s); until then the first failing check is shown as the hint. The camera's tilt
// (accelerometer) is one more check: 'level' when it exceeds `maxCameraTiltDeg`.
import 'dart:math' as math;

import 'capture_qc.dart';

class AutoCaptureState {
  const AutoCaptureState({required this.hint, required this.streak, required this.fire});

  /// The hint to show ('ready' while the streak builds), in `hintOrder` priority.
  final String hint;
  final int streak;

  /// True once: take the photo now.
  final bool fire;
}

/// How far the corners moved between two frames, as a share of the sheet's diagonal.
double cornerJitter(List<List<double>> a, List<List<double>> b) {
  var diag = 0.0;
  for (var i = 0; i < 4; i++) {
    for (var j = i + 1; j < 4; j++) {
      diag = math.max(diag, math.sqrt(math.pow(a[i][0] - a[j][0], 2) + math.pow(a[i][1] - a[j][1], 2)));
    }
  }
  if (diag <= 0) return double.infinity;
  var move = 0.0;
  for (var i = 0; i < 4; i++) {
    move = math.max(move, math.sqrt(math.pow(a[i][0] - b[i][0], 2) + math.pow(a[i][1] - b[i][1], 2)));
  }
  return move / diag;
}

class AutoCapture {
  AutoCapture(this.consecutiveFrames,
      {this.stableMs = 0, this.maxJitter = double.infinity, this.maxCameraTiltDeg = double.infinity});

  /// From the `auto_capture` and `thresholds` sections of the frame-check config.
  factory AutoCapture.fromConfig(Map<String, dynamic>? config) {
    final a = (config?['auto_capture'] as Map?) ?? const {};
    final t = (config?['thresholds'] as Map?) ?? const {};
    return AutoCapture((a['consecutive_frames'] as num? ?? 5).toInt(),
        stableMs: (a['stable_ms'] as num? ?? 0).toInt(),
        maxJitter: (a['corner_jitter'] as num?)?.toDouble() ?? double.infinity,
        maxCameraTiltDeg: (t['max_camera_tilt_deg'] as num?)?.toDouble() ?? double.infinity);
  }

  final int consecutiveFrames;
  final int stableMs;
  final double maxJitter;
  final double maxCameraTiltDeg;
  int _streak = 0;
  bool _fired = false;
  DateTime? _stableSince;
  List<List<double>>? _lastCorners;

  /// The hints of `qc` plus the camera's tilt, in display order.
  List<String> hintsFor(FrameQc qc, double? cameraTiltDeg) {
    final h = {...qc.hints};
    if (cameraTiltDeg != null && cameraTiltDeg > maxCameraTiltDeg) h.add('level');
    return hintOrder.where(h.contains).toList();
  }

  AutoCaptureState update(FrameQc qc, {DateTime? now, double? cameraTiltDeg}) {
    final t = now ?? DateTime.now();
    final hints = hintsFor(qc, cameraTiltDeg);
    if (hints.isNotEmpty) {
      reset(keepFired: true);
      return AutoCaptureState(hint: hints.first, streak: 0, fire: false);
    }
    final c = qc.corners;
    final last = _lastCorners;
    if (c == null || last == null || cornerJitter(last, c) > maxJitter) _stableSince = t;
    _stableSince ??= t;
    _lastCorners = c;
    _streak++;
    final steady = t.difference(_stableSince!).inMilliseconds >= stableMs;
    final fire = !_fired && _streak >= consecutiveFrames && steady;
    if (fire) _fired = true;
    return AutoCaptureState(hint: 'ready', streak: _streak, fire: fire);
  }

  void reset({bool keepFired = false}) {
    _streak = 0;
    _stableSince = null;
    _lastCorners = null;
    if (!keepFired) _fired = false;
  }
}
