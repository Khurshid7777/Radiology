// After the shot (TZ addendum §18.3): the sheet's four corners, found by the frame
// checks on the photo itself, to drag where the doctor sees the sheet; a magnifier
// (3x) under the finger; turn by 90 degrees; retake; done.
//
// Nothing here changes the photo (§18.1): the corners (in the upright photo's pixels)
// and the turn travel with it as the capture record; the server straightens the page by
// the grid itself and uses the turn before reading.
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import '../qc/capture_qc.dart';
import '../state/app_state.dart';

class CropResult {
  CropResult(
      {required this.auto,
      required this.corners,
      required this.edited,
      required this.rotateDeg,
      this.width = 0,
      this.height = 0});

  final List<List<double>>? auto;
  final List<List<double>> corners;
  final bool edited;

  /// Clockwise turn the server applies before reading (0, 90, 180, 270).
  final int rotateDeg;

  /// The upright photo's size in pixels (the corners' frame); 0 when unknown.
  final int width;
  final int height;

  Map<String, dynamic> toCaptureCorners() => {'auto': auto, 'final': corners, 'edited_by_user': edited};
}

/// What the crop screen needs from the photo: its upright size and the sheet's corners.
class PhotoSheet {
  PhotoSheet(this.width, this.height, this.corners);

  final int width;
  final int height;
  final List<List<double>>? corners;
}

/// Decodes the photo (EXIF orientation applied) and finds the sheet on a <= 1280-px copy.
PhotoSheet findSheet((Uint8List, Map<String, dynamic>) a) {
  final photo = img.decodeImage(a.$1);
  if (photo == null) throw ArgumentError('not a decodable image');
  final up = img.bakeOrientation(photo);
  final small = up.width > 1280 ? img.copyResize(up, width: 1280, interpolation: img.Interpolation.average) : up;
  final qc = checkFrame(small.getBytes(order: img.ChannelOrder.rgb), small.width, small.height, a.$2,
      scaleToCapture: up.width / small.width);
  return PhotoSheet(up.width, up.height, qc.corners);
}

/// The whole photo, when no sheet was found: the doctor drags the corners in.
List<List<double>> wholePhoto(int w, int h) => [
      [0, 0],
      [w.toDouble(), 0],
      [w.toDouble(), h.toDouble()],
      [0, h.toDouble()],
    ];

class CropScreen extends StatefulWidget {
  const CropScreen({super.key, required this.app, required this.photo, this.sheet});

  final AppState app;
  final Uint8List photo;

  /// Precomputed (tests); otherwise found in a background isolate.
  final PhotoSheet? sheet;

  @override
  State<CropScreen> createState() => _CropScreenState();
}

class _CropScreenState extends State<CropScreen> {
  PhotoSheet? _sheet;
  List<List<double>>? _corners;
  bool _edited = false;
  int _turns = 0;
  int? _dragging;
  Offset? _magnifierAt;
  final _outer = GlobalKey();

  @override
  void initState() {
    super.initState();
    if (widget.sheet != null) {
      _ready(widget.sheet!);
    } else {
      compute(findSheet, (widget.photo, widget.app.qcConfig ?? const <String, dynamic>{})).then(_ready,
          onError: (Object _) => _ready(PhotoSheet(1, 1, null)));
    }
  }

  void _ready(PhotoSheet s) {
    if (!mounted) return;
    setState(() {
      _sheet = s;
      _corners = [for (final c in s.corners ?? wholePhoto(s.width, s.height)) [...c]];
    });
  }

  void _drag(int i, DragUpdateDetails d, double scale) {
    final s = _sheet!;
    final c = _corners![i];
    setState(() {
      c[0] = (c[0] + d.delta.dx / scale).clamp(0, s.width.toDouble());
      c[1] = (c[1] + d.delta.dy / scale).clamp(0, s.height.toDouble());
      _edited = true;
      final box = _outer.currentContext?.findRenderObject() as RenderBox?;
      _magnifierAt = box?.globalToLocal(d.globalPosition);
    });
  }

  Widget _photo(BoxConstraints box) {
    final s = _sheet!;
    final scale = [box.maxWidth / s.width, box.maxHeight / s.height].reduce((a, b) => a < b ? a : b);
    const r = 22.0;
    return SizedBox(
      width: s.width * scale,
      height: s.height * scale,
      child: Stack(clipBehavior: Clip.none, children: [
        Positioned.fill(child: Image.memory(widget.photo, fit: BoxFit.fill, gaplessPlayback: true)),
        Positioned.fill(child: CustomPaint(painter: _QuadPainter(_corners!, scale))),
        for (var i = 0; i < 4; i++)
          Positioned(
            left: _corners![i][0] * scale - r,
            top: _corners![i][1] * scale - r,
            child: GestureDetector(
              key: Key('corner_$i'),
              behavior: HitTestBehavior.opaque,
              onPanStart: (_) => setState(() => _dragging = i),
              onPanUpdate: (d) => _drag(i, d, scale),
              onPanEnd: (_) => setState(() {
                _dragging = null;
                _magnifierAt = null;
              }),
              child: Container(
                width: 2 * r,
                height: 2 * r,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: (_dragging == i ? Colors.greenAccent : Colors.amberAccent).withValues(alpha: 0.35),
                  border: Border.all(color: Colors.white, width: 2),
                ),
              ),
            ),
          ),
      ]),
    );
  }

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    final s = _sheet;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.crop_title'))),
      body: Column(children: [
        Padding(padding: const EdgeInsets.all(12), child: Text(app.t('app.crop_hint'))),
        Expanded(
          child: s == null
              ? const Center(child: CircularProgressIndicator())
              : Stack(key: _outer, children: [
                  Center(
                    child: RotatedBox(
                      quarterTurns: _turns,
                      child: LayoutBuilder(builder: (context, box) => Center(child: _photo(box))),
                    ),
                  ),
                  if (_magnifierAt != null)
                    Positioned(
                      left: _magnifierAt!.dx - 60,
                      top: _magnifierAt!.dy - 60 - 90,
                      child: const RawMagnifier(
                        size: Size(120, 120),
                        magnificationScale: 3,
                        focalPointOffset: Offset(0, 90),
                        decoration: MagnifierDecoration(
                            shape: CircleBorder(side: BorderSide(color: Colors.white, width: 3))),
                      ),
                    ),
                ]),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(children: [
            Expanded(
                child: OutlinedButton(
                    key: const Key('crop_retake'), onPressed: () => Navigator.pop(context), child: Text(app.t('app.retake')))),
            const SizedBox(width: 8),
            IconButton.outlined(
              key: const Key('crop_rotate'),
              tooltip: app.t('app.rotate'),
              icon: const Icon(Icons.rotate_90_degrees_cw),
              onPressed: s == null ? null : () => setState(() => _turns = (_turns + 1) % 4),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: FilledButton(
                key: const Key('crop_done'),
                onPressed: s == null
                    ? null
                    : () => Navigator.pop(
                        context,
                        CropResult(
                            auto: s.corners,
                            corners: _corners!,
                            edited: _edited,
                            rotateDeg: 90 * _turns,
                            width: s.width,
                            height: s.height)),
                child: Text(app.t('app.done')),
              ),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _QuadPainter extends CustomPainter {
  _QuadPainter(this.corners, this.scale);

  final List<List<double>> corners;
  final double scale;

  @override
  void paint(Canvas canvas, Size size) {
    final path = Path()..moveTo(corners[0][0] * scale, corners[0][1] * scale);
    for (final c in corners.skip(1)) {
      path.lineTo(c[0] * scale, c[1] * scale);
    }
    path.close();
    canvas.drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Colors.amberAccent);
  }

  @override
  bool shouldRepaint(_QuadPainter old) => true;
}
