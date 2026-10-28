// The doctor's confirmation screen (TZ §6.3): the extracted traces over the photo,
// speed and gain, the status in plain words; accept, correct and re-run, or retake.
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../api/scan_api.dart';
import '../state/app_state.dart';
import 'camera_screen.dart';
import 'widgets.dart';

class ConfirmScreen extends StatefulWidget {
  const ConfirmScreen({super.key, required this.app, required this.scanId});

  final AppState app;
  final String scanId;

  @override
  State<ConfirmScreen> createState() => _ConfirmScreenState();
}

class _ConfirmScreenState extends State<ConfirmScreen> {
  Map<String, dynamic>? _scan;
  Uint8List? _overlay;
  double? _speed;
  double? _gain;
  String? _layout;
  String? _error;
  String? _conclusion;
  bool _working = false;

  AppState get app => widget.app;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final scan = await app.api.get(widget.scanId);
      final cal = (scan['result'] as Map<String, dynamic>?)?['calibration'] as Map<String, dynamic>?;
      Uint8List? overlay;
      try {
        overlay = await app.api.overlay(widget.scanId);
      } on ApiError catch (_) {}
      setState(() {
        _scan = scan;
        _overlay = overlay;
        _speed = (cal?['paper_speed_mm_s'] as num?)?.toDouble();
        _gain = (cal?['gain_mm_mv'] as num?)?.toDouble();
        _layout = ((scan['result'] as Map<String, dynamic>?)?['layout'] as Map<String, dynamic>?)?['name'] as String?;
        _error = null;
      });
    } on ApiError catch (e) {
      setState(() => _error = e.message);
    }
  }

  Future<void> _act(Future<Map<String, dynamic>> Function() call, {bool reload = true}) async {
    setState(() => _working = true);
    try {
      await call();
      if (reload) await _load();
    } on ApiError catch (e) {
      setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _working = false);
    }
  }

  Widget _panel(Map<String, dynamic> scan) {
    final status = scan['status'] as String;
    final msg = (scan['messages'] as Map<String, dynamic>)[app.settings.lang] as Map<String, dynamic>;
    final result = scan['result'] as Map<String, dynamic>?;
    final cal = result?['calibration'] as Map<String, dynamic>?;
    final found = (result?['layout'] as Map<String, dynamic>?)?['name'] as String?;
    final layoutChanged = _layout != null && _layout != found;
    final changed = cal != null && (_speed != (cal['paper_speed_mm_s'] as num?)?.toDouble() ||
            _gain != (cal['gain_mm_mv'] as num?)?.toDouble()) ||
        layoutChanged;
    final pending = const ['queued', 'processing'].contains(status);
    final reasons = ((result?['reject_reasons'] as List?) ?? const []).cast<String>();
    final warnings = ((result?['warnings'] as List?) ?? const []).cast<String>();
    // TZ 18.5: when the layout is unsure or unknown the doctor names it, the server does not guess.
    final askLayout = reasons.contains('LAYOUT_UNKNOWN') || warnings.contains('LAYOUT_UNCERTAIN');
    return ListView(padding: const EdgeInsets.all(16), children: [
      Text(msg['status'] as String, style: TextStyle(fontSize: 18, color: statusColor(status))),
      for (final r in (msg['reasons'] as List).cast<String>()) Text('• $r'),
      for (final w in (msg['warnings'] as List).cast<String>()) Text('• $w'),
      const SizedBox(height: 12),
      if (result != null && (status != 'rejected' || reasons.contains('LAYOUT_UNKNOWN'))) ...[
        if (askLayout) Text(app.t('app.layout_ask'), key: const Key('layout_ask')),
        LayoutChips(label: app.t('app.layout'), value: _layout, onChanged: (v) => setState(() => _layout = v)),
      ],
      if (result != null && status == 'rejected' && reasons.contains('LAYOUT_UNKNOWN'))
        OutlinedButton(
          key: const Key('correct_layout'),
          onPressed: _working || pending || !layoutChanged
              ? null
              : () => _act(() => app.api.confirm(widget.scanId, 'corrected', layout: _layout)),
          child: Text(app.t('app.correct')),
        ),
      if (result != null && status != 'rejected') ...[
        Text(app.t('app.overlay_hint')),
        const SizedBox(height: 8),
        ValueChips(label: app.t('app.speed'), values: const [25, 50], value: _speed, unknown: app.t('app.unknown'),
            unit: app.settings.lang == 'uz' ? 'mm/s' : 'мм/с', onChanged: (v) => setState(() => _speed = v)),
        ValueChips(label: app.t('app.gain'), values: const [5, 10, 20], value: _gain, unknown: app.t('app.unknown'),
            unit: app.settings.lang == 'uz' ? 'mm/mV' : 'мм/мВ', onChanged: (v) => setState(() => _gain = v)),
        const SizedBox(height: 12),
        Wrap(spacing: 8, runSpacing: 8, children: [
          FilledButton(
            key: const Key('accept'),
            onPressed: _working || pending || changed || _speed == null || _gain == null
                ? null
                : () => _act(() => app.api.confirm(widget.scanId, 'accepted')),
            child: Text(app.t('app.accept')),
          ),
          OutlinedButton(
            key: const Key('correct'),
            onPressed: _working || pending || !changed
                ? null
                : () => _act(() => app.api.confirm(widget.scanId, 'corrected',
                    paperSpeedMmS: _speed, gainMmMv: _gain, layout: layoutChanged ? _layout : null)),
            child: Text(app.t('app.correct')),
          ),
        ]),
      ],
      const SizedBox(height: 8),
      OutlinedButton(
        key: const Key('retake'),
        onPressed: _working || pending
            ? null
            : () async {
                if (result != null) await _act(() => app.api.confirm(widget.scanId, 'retake'), reload: false);
                if (!mounted) return;
                Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => CameraScreen(app: app)));
              },
        child: Text(app.t('app.retake')),
      ),
      if (status == 'ok') ...[
        const SizedBox(height: 16),
        OutlinedButton(
          key: const Key('analysis'),
          onPressed: _working
              ? null
              : () => _act(() async {
                    final a = await app.api.analyze(widget.scanId);
                    setState(() => _conclusion = a['conclusion_${app.settings.lang}'] as String?);
                    return a;
                  }, reload: false),
          child: Text(app.t('app.analysis')),
        ),
        if (_conclusion != null) Padding(padding: const EdgeInsets.only(top: 8), child: SelectableText(_conclusion!)),
      ],
    ]);
  }

  @override
  Widget build(BuildContext context) {
    final scan = _scan;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.confirm_title')),
          actions: [IconButton(icon: const Icon(Icons.refresh), onPressed: _working ? null : _load)]),
      body: _error != null && scan == null
          ? Center(child: Text(app.t('app.error', {'message': _error!})))
          : scan == null
              ? const Center(child: CircularProgressIndicator())
              : LayoutBuilder(builder: (context, box) {
                  final image = _overlay == null
                      ? const SizedBox.shrink()
                      : InteractiveViewer(maxScale: 8, child: Image.memory(_overlay!, fit: BoxFit.contain));
                  // Tablet: the overlay beside the panel; phone: above it (TZ §6.4).
                  return box.maxWidth >= 840
                      ? Row(children: [Expanded(flex: 3, child: image), SizedBox(width: 380, child: _panel(scan))])
                      : Column(children: [
                          if (_overlay != null) SizedBox(height: box.maxHeight * 0.45, child: image),
                          Expanded(child: _panel(scan)),
                        ]);
                }),
    );
  }
}
