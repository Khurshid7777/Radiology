// The camera screen (TZ §6.1): live frame checks with RU/UZ hints, auto-capture when
// every check passes several frames in a row, a manual shutter, speed/gain chips.
import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../camera/frame_convert.dart';
import '../qc/auto_capture.dart';
import '../qc/capture_qc.dart';
import '../state/app_state.dart';
import 'mask_review_screen.dart';
import 'widgets.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.app});

  final AppState app;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cam;
  String? _error;
  bool _busy = false;
  bool _taking = false;
  String _hint = 'frame';
  double? _speed;
  double? _gain;
  late AutoCapture _auto;

  Map<String, dynamic>? get _qc => widget.app.qcConfig;

  @override
  void initState() {
    super.initState();
    _auto = AutoCapture(((_qc?['auto_capture'] as Map?)?['consecutive_frames'] as num? ?? 5).toInt());
    _open();
  }

  Future<void> _open() async {
    try {
      final cams = await availableCameras();
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      // Full resolution: a whole page needs >= 8 px/mm in the photo (TZ §6.1).
      final cam = CameraController(back, ResolutionPreset.max, enableAudio: false,
          imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420);
      await cam.initialize();
      if (_qc != null) await cam.startImageStream(_onFrame);
      setState(() => _cam = cam);
    } on Object catch (e) {
      setState(() => _error = e.toString());
    }
  }

  void _onFrame(CameraImage image) {
    if (_busy || _taking || _qc == null) return;
    _busy = true;
    final target = ((_qc!['algorithm'] as Map)['work_width'] as num).toInt();
    final RgbFrame frame;
    if (image.format.group == ImageFormatGroup.bgra8888) {
      final p = image.planes.first;
      frame = bgraToRgb(width: image.width, height: image.height, bytes: p.bytes, rowStride: p.bytesPerRow,
          targetWidth: target);
    } else {
      frame = yuv420ToRgb(width: image.width, height: image.height, y: image.planes[0].bytes,
          yRowStride: image.planes[0].bytesPerRow, u: image.planes[1].bytes, v: image.planes[2].bytes,
          uvRowStride: image.planes[1].bytesPerRow, uvPixelStride: image.planes[1].bytesPerPixel ?? 1,
          targetWidth: target);
    }
    compute(_check, (frame, _qc!)).then((qc) {
      if (!mounted) return;
      final s = _auto.update(qc);
      setState(() => _hint = s.hint);
      if (s.fire && widget.app.settings.autoCapture) _take();
    }).whenComplete(() => _busy = false);
  }

  Future<void> _take() async {
    final cam = _cam;
    if (cam == null || _taking) return;
    setState(() => _taking = true);
    try {
      if (cam.value.isStreamingImages) await cam.stopImageStream();
      final shot = await cam.takePicture();
      final bytes = await shot.readAsBytes();
      double? band;
      if (widget.app.settings.maskHeader) {
        if (!mounted) return;
        band = await Navigator.push<double>(
            context, MaterialPageRoute(builder: (_) => MaskReviewScreen(app: widget.app, photo: bytes)));
        if (band == null) {
          // Retake: back to the live preview.
          _auto.reset();
          if (_qc != null) await cam.startImageStream(_onFrame);
          setState(() => _taking = false);
          return;
        }
      }
      await widget.app.addPhoto(bytes, speed: _speed, gain: _gain, band: band);
      if (mounted) Navigator.pop(context);
    } on Object catch (e) {
      setState(() {
        _error = e.toString();
        _taking = false;
      });
    }
  }

  @override
  void dispose() {
    _cam?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    final cam = _cam;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.capture'))),
      body: Column(children: [
        Expanded(
          child: _error != null
              ? Center(child: Text(app.t('app.error', {'message': _error!})))
              : cam == null
                  ? const Center(child: CircularProgressIndicator())
                  : Stack(fit: StackFit.expand, children: [
                      CameraPreview(cam),
                      // The frame the sheet should fill.
                      IgnorePointer(
                        child: Container(
                          margin: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                              border: Border.all(color: _hint == 'ready' ? Colors.green : Colors.white70, width: 3)),
                        ),
                      ),
                      if (_qc != null)
                        Positioned(
                          left: 16,
                          right: 16,
                          bottom: 16,
                          child: Container(
                            padding: const EdgeInsets.all(12),
                            color: Colors.black54,
                            child: Text(app.t('qc.$_hint'),
                                textAlign: TextAlign.center, style: const TextStyle(color: Colors.white, fontSize: 18)),
                          ),
                        ),
                    ]),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ValueChips(label: app.t('app.speed'), values: const [25, 50], value: _speed, unknown: app.t('app.unknown'),
                unit: app.settings.lang == 'uz' ? 'mm/s' : 'мм/с', onChanged: (v) => setState(() => _speed = v)),
            ValueChips(label: app.t('app.gain'), values: const [5, 10, 20], value: _gain, unknown: app.t('app.unknown'),
                unit: app.settings.lang == 'uz' ? 'mm/mV' : 'мм/мВ', onChanged: (v) => setState(() => _gain = v)),
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                icon: const Icon(Icons.camera),
                label: Text(app.t('app.manual_shot')),
                onPressed: cam == null || _taking ? null : _take,
              ),
            ),
          ]),
        ),
      ]),
    );
  }
}

FrameQc _check((RgbFrame, Map<String, dynamic>) a) =>
    checkFrame(a.$1.rgb, a.$1.width, a.$1.height, a.$2, scaleToCapture: a.$1.step.toDouble());
