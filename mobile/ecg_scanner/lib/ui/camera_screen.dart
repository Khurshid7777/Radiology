// The camera screen (TZ §6.1, addendum §18.2): live frame checks with RU/UZ hints, the
// sheet found drawn over the preview, the phone's tilt from the accelerometer,
// auto-capture when every check passes and the corners hold still ~0.7 s, a manual
// shutter, speed/gain chips. The flash is off (it glares on thermal paper); the torch is
// offered only when the frame is too dark. After the shot: the crop screen (§18.3).
import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';

import '../camera/frame_convert.dart';
import '../camera/sheet_overlay.dart';
import '../qc/auto_capture.dart';
import '../qc/camera_tilt.dart';
import '../qc/capture_qc.dart';
import '../state/app_state.dart';
import 'crop_screen.dart';
import 'mask_review_screen.dart';
import 'widgets.dart';

/// The phone's frame checks as the capture record carries them (ecgscan.contract.CaptureQC).
Map<String, dynamic> qcRecord(FrameQc qc, List<String> hints) => {
      'brightness': qc.brightness,
      'glare': qc.glare,
      'sharpness': qc.sharpness,
      'shadow': qc.shadow,
      'px_per_mm': qc.pxPerMm,
      'tilt_deg': qc.tiltDeg,
      'sheet_area': qc.sheetArea,
      'passed': hints.isEmpty,
      'failed_checks': hints,
    };

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.app, this.multi = false});

  final AppState app;

  /// A sheet of a study on several pages (§18.4): kept in the draft, not queued on its own.
  final bool multi;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cam;
  String? _error;
  bool _busy = false;
  bool _taking = false;
  bool _torch = false;
  String _hint = 'frame';
  double? _speed;
  double? _gain;
  FrameQc? _lastQc;
  List<String> _lastHints = const ['frame'];
  List<Offset>? _quad;
  late AutoCapture _auto;
  final _tilt = TiltFilter();
  StreamSubscription<AccelerometerEvent>? _accel;

  Map<String, dynamic>? get _qc => widget.app.qcConfig;

  @override
  void initState() {
    super.initState();
    _auto = AutoCapture.fromConfig(_qc);
    // The overlay maps the sensor's frames onto a portrait preview.
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    try {
      _accel = accelerometerEventStream(samplingPeriod: SensorInterval.uiInterval)
          .listen((e) => _tilt.update(e.x, e.y, e.z), onError: (Object _) {});
    } on Object catch (_) {
      // No accelerometer (emulator, tests): the tilt check is simply absent.
    }
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
      await cam.setFlashMode(FlashMode.off);
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
    final orientation = _cam?.description.sensorOrientation ?? 90;
    compute(_check, (frame, _qc!)).then((qc) {
      if (!mounted) return;
      final tilt = _tilt.value;
      final s = _auto.update(qc, cameraTiltDeg: tilt);
      setState(() {
        _hint = s.hint;
        _lastQc = qc;
        _lastHints = _auto.hintsFor(qc, tilt);
        _quad = previewQuad(qc.corners, image.width.toDouble(), image.height.toDouble(), orientation);
      });
      if (s.fire && widget.app.settings.autoCapture) _take(auto: true);
    }).whenComplete(() => _busy = false);
  }

  Future<void> _resume() async {
    _auto.reset();
    final cam = _cam;
    if (cam != null && _qc != null && !cam.value.isStreamingImages) await cam.startImageStream(_onFrame);
    if (mounted) setState(() => _taking = false);
  }

  Future<void> _setTorch(bool on) async {
    await _cam?.setFlashMode(on ? FlashMode.torch : FlashMode.off);
    setState(() => _torch = on);
  }

  Future<void> _take({bool auto = false}) async {
    final cam = _cam;
    if (cam == null || _taking) return;
    setState(() => _taking = true);
    final qcAtShot = _lastQc;
    final hintsAtShot = _lastHints;
    final tiltAtShot = _tilt.value;
    try {
      if (cam.value.isStreamingImages) await cam.stopImageStream();
      final shot = await cam.takePicture();
      final bytes = await shot.readAsBytes();
      // The camera's temporary file is not kept: photos live only in the encrypted queue (§18.9).
      try {
        await File(shot.path).delete();
      } on Object catch (_) {}
      if (_torch) await _setTorch(false);
      if (!mounted) return;
      final crop = await Navigator.push<CropResult>(
          context, MaterialPageRoute(builder: (_) => CropScreen(app: widget.app, photo: bytes)));
      if (crop == null) return await _resume();
      double? band;
      if (widget.app.settings.maskHeader) {
        if (!mounted) return;
        band = await Navigator.push<double>(
            context, MaterialPageRoute(builder: (_) => MaskReviewScreen(app: widget.app, photo: bytes)));
        if (band == null) return await _resume();
      }
      final capture = <String, dynamic>{
        'capture_mode': 'custom_camera',
        'auto_capture': auto,
        'corners': crop.toCaptureCorners(),
        'rotate_deg': crop.rotateDeg,
        if (tiltAtShot != null) 'camera_tilt_deg': double.parse(tiltAtShot.clamp(0, 90).toStringAsFixed(1)),
        if (qcAtShot != null) 'qc': qcRecord(qcAtShot, hintsAtShot),
      };
      if (widget.multi) {
        await widget.app.addSheet(bytes, band: band, capture: capture);
      } else {
        await widget.app.addPhoto(bytes, speed: _speed, gain: _gain, band: band, capture: capture);
      }
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
    _accel?.cancel();
    _cam?.dispose();
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    final cam = _cam;
    final dark = _hint == 'light' || _torch;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.capture'))),
      body: Column(children: [
        Expanded(
          child: _error != null
              ? Center(child: Text(app.t('app.error', {'message': _error!})))
              : cam == null
                  ? const Center(child: CircularProgressIndicator())
                  : Stack(fit: StackFit.expand, children: [
                      CameraPreview(cam,
                          child: IgnorePointer(
                              child: CustomPaint(painter: SheetOverlayPainter(_quad, ready: _hint == 'ready')))),
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
                      if (dark)
                        Positioned(
                          top: 12,
                          right: 12,
                          child: FilledButton.tonalIcon(
                            key: const Key('torch'),
                            icon: Icon(_torch ? Icons.flashlight_off : Icons.flashlight_on),
                            label: Text(app.t('app.torch')),
                            onPressed: () => _setTorch(!_torch),
                          ),
                        ),
                    ]),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            if (!widget.multi) ...[
              ValueChips(label: app.t('app.speed'), values: const [25, 50], value: _speed, unknown: app.t('app.unknown'),
                  unit: app.settings.lang == 'uz' ? 'mm/s' : 'мм/с', onChanged: (v) => setState(() => _speed = v)),
              ValueChips(label: app.t('app.gain'), values: const [5, 10, 20], value: _gain, unknown: app.t('app.unknown'),
                  unit: app.settings.lang == 'uz' ? 'mm/mV' : 'мм/мВ', onChanged: (v) => setState(() => _gain = v)),
              const SizedBox(height: 8),
            ],
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                icon: const Icon(Icons.camera),
                label: Text(app.t('app.manual_shot')),
                onPressed: cam == null || _taking ? null : () => _take(),
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
