// Before upload: what will be masked (TZ §6.2, addendum §18.6; S3). With the recorder's
// profile, its personal-data zones are shown on the sheet the doctor framed, and the band
// starts empty: the doctor adds it only if something is left uncovered (the manual
// correction). Without a profile, the top band as before; its edge is moved to cover the
// name and ID but not the traces. Returns the band, or null to retake.
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../masking/zone_mask.dart';
import '../state/app_state.dart';
import 'crop_screen.dart';

class MaskReviewScreen extends StatefulWidget {
  const MaskReviewScreen({super.key, required this.app, required this.photo, this.crop});

  final AppState app;
  final Uint8List photo;

  /// The confirmed sheet (corners, quarter turns, photo size): needed to place the zones.
  final CropResult? crop;

  @override
  State<MaskReviewScreen> createState() => _MaskReviewScreenState();
}

class _MaskReviewScreenState extends State<MaskReviewScreen> {
  bool get _zones => widget.app.settings.hasZones && widget.crop != null && widget.crop!.width > 0;

  late double _band = _zones ? 0.0 : widget.app.settings.headerBand;

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    final s = app.settings;
    final crop = widget.crop;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.mask_header'))),
      body: Column(children: [
        Padding(
            padding: const EdgeInsets.all(12),
            child: Text(_zones ? app.t('app.mask_review_zones', {'device': s.deviceProfileName ?? ''}) : app.t('app.mask_review'))),
        Expanded(
          child: Center(
            child: Stack(children: [
              Image.memory(widget.photo, fit: BoxFit.contain),
              Positioned.fill(
                child: LayoutBuilder(builder: (context, box) {
                  return Stack(children: [
                    if (_zones)
                      Positioned.fill(
                        child: CustomPaint(
                          key: const Key('zones'),
                          painter: _ZonesPainter([
                            for (final z in s.deviceZones)
                              zonePolygon(z, crop!.corners, crop.rotateDeg, margin: s.zoneMargin)
                          ], box.maxWidth / crop!.width),
                        ),
                      ),
                    Align(
                      alignment: Alignment.topCenter,
                      child: Container(height: box.maxHeight * _band, color: Colors.white.withValues(alpha: 0.9)),
                    ),
                  ]);
                }),
              ),
            ]),
          ),
        ),
        Slider(key: const Key('band'), value: _band, min: 0.0, max: 0.35, onChanged: (v) => setState(() => _band = v)),
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
          child: Row(children: [
            Expanded(child: OutlinedButton(onPressed: () => Navigator.pop(context), child: Text(app.t('app.retake')))),
            const SizedBox(width: 12),
            Expanded(
              child: FilledButton(
                key: const Key('send'),
                onPressed: () => Navigator.pop(context, _band),
                child: Text(app.t('app.send')),
              ),
            ),
          ]),
        ),
      ]),
    );
  }
}

class _ZonesPainter extends CustomPainter {
  _ZonesPainter(this.polygons, this.scale);

  final List<List<List<double>>> polygons;
  final double scale;

  @override
  void paint(Canvas canvas, Size size) {
    final fill = Paint()..color = Colors.white.withValues(alpha: 0.9);
    final edge = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    for (final poly in polygons) {
      final path = Path()..moveTo(poly[0][0] * scale, poly[0][1] * scale);
      for (final p in poly.skip(1)) {
        path.lineTo(p[0] * scale, p[1] * scale);
      }
      path.close();
      canvas
        ..drawPath(path, fill)
        ..drawPath(path, edge);
    }
  }

  @override
  bool shouldRepaint(_ZonesPainter old) => old.polygons != polygons || old.scale != scale;
}
