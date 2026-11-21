// SD-133: where to move the phone when the sheet leaves the frame. The rule table is the Python one
// (tests/scanner/test_capture_qc.py); the direction is turned to the preview like the sheet corners.
import 'package:ecg_scanner/camera/sheet_overlay.dart';
import 'package:ecg_scanner/qc/capture_qc.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('rule table: top, bottom, left, right blocks against the edge threshold', () {
    expect(frameAdvice([0, 0, 0, 0], 3), isNull);
    expect(frameAdvice([2, 2, 2, 2], 3), isNull);
    expect(frameAdvice([5, 0, 0, 0], 3), 'up');
    expect(frameAdvice([0, 4, 0, 0], 3), 'down');
    expect(frameAdvice([0, 0, 2, 9], 3), 'right');
    expect(frameAdvice([4, 0, 7, 0], 3), 'left');
    expect(frameAdvice([5, 4, 0, 0], 3), 'farther');
    expect(frameAdvice([0, 0, 3, 3], 3), 'farther');
  });

  test('the advice is turned to the upright preview like the corners', () {
    expect(turnAdvice(null, 90), isNull);
    expect(turnAdvice('farther', 90), 'farther');
    for (final a in ['up', 'down', 'left', 'right']) {
      expect(turnAdvice(a, 0), a);
    }
    // sensor at 90 degrees: frame (x, y) -> preview (1 - y, x)
    expect(turnAdvice('left', 90), 'up');
    expect(turnAdvice('right', 90), 'down');
    expect(turnAdvice('up', 90), 'right');
    expect(turnAdvice('down', 90), 'left');
    expect(turnAdvice('left', 180), 'right');
    expect(turnAdvice('up', 180), 'down');
    expect(turnAdvice('left', 270), 'down');
    expect(turnAdvice('up', 270), 'left');
  });
}
