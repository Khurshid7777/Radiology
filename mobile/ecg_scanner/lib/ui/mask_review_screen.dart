// Before upload: the photo with the masking band (TZ §6.2). The band covers the top
// of the sheet by default; headers differ between machines, so the doctor moves its
// edge to cover the name and ID but not the traces. Returns the band, or null to retake.
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../state/app_state.dart';

class MaskReviewScreen extends StatefulWidget {
  const MaskReviewScreen({super.key, required this.app, required this.photo});

  final AppState app;
  final Uint8List photo;

  @override
  State<MaskReviewScreen> createState() => _MaskReviewScreenState();
}

class _MaskReviewScreenState extends State<MaskReviewScreen> {
  late double _band = widget.app.settings.headerBand;

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.mask_header'))),
      body: Column(children: [
        Padding(padding: const EdgeInsets.all(12), child: Text(app.t('app.mask_review'))),
        Expanded(
          child: Center(
            child: Stack(children: [
              Image.memory(widget.photo, fit: BoxFit.contain),
              Positioned.fill(
                child: LayoutBuilder(builder: (context, box) {
                  return Align(
                    alignment: Alignment.topCenter,
                    child: Container(height: box.maxHeight * _band, color: Colors.white.withValues(alpha: 0.9)),
                  );
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
