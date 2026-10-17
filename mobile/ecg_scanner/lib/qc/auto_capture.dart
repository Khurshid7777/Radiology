// Auto-capture (TZ §6.1): the shot is taken when every check passes on several
// consecutive frames; until then the first failing check is shown as the hint.
import 'capture_qc.dart';

class AutoCaptureState {
  const AutoCaptureState({required this.hint, required this.streak, required this.fire});

  /// The hint to show ('ready' while the streak builds), in `hintOrder` priority.
  final String hint;
  final int streak;

  /// True once: take the photo now.
  final bool fire;
}

class AutoCapture {
  AutoCapture(this.consecutiveFrames);

  final int consecutiveFrames;
  int _streak = 0;
  bool _fired = false;

  AutoCaptureState update(FrameQc qc) {
    if (!qc.ok) {
      _streak = 0;
      return AutoCaptureState(hint: hintOrder.firstWhere(qc.hints.contains), streak: 0, fire: false);
    }
    _streak++;
    final fire = !_fired && _streak >= consecutiveFrames;
    if (fire) _fired = true;
    return AutoCaptureState(hint: 'ready', streak: _streak, fire: fire);
  }

  void reset() {
    _streak = 0;
    _fired = false;
  }
}
