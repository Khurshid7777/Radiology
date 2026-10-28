// A study on several sheets (TZ addendum §18.4): the sheets taken so far in the order they go
// to the server, dragged to reorder, deleted, one more taken; then the study is sent.
// The draft is on disk: leaving the app never loses a sheet.
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../state/app_state.dart';
import 'camera_screen.dart';
import 'widgets.dart';

class SheetsScreen extends StatefulWidget {
  const SheetsScreen({super.key, required this.app});

  final AppState app;

  @override
  State<SheetsScreen> createState() => _SheetsScreenState();
}

class _SheetsScreenState extends State<SheetsScreen> {
  double? _speed;
  double? _gain;
  bool _working = false;

  AppState get app => widget.app;

  Future<void> _more() async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => CameraScreen(app: app, multi: true)));
    if (mounted) setState(() {});
  }

  Future<void> _run(Future<void> Function() step) async {
    setState(() => _working = true);
    try {
      await step();
    } finally {
      if (mounted) setState(() => _working = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final draft = app.queue.draft;
    final ids = draft?.partIds ?? const <String>[];
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.sheets_title'))),
      body: Column(children: [
        Padding(padding: const EdgeInsets.all(12), child: Text(app.t('app.sheets_hint'))),
        Expanded(
          child: ids.isEmpty
              ? Center(child: Text(app.t('app.sheets_empty')))
              : ReorderableListView(
                  key: const Key('sheets'),
                  onReorderItem: (from, to) => _run(() => app.queue.moveSheet(draft!, from, to)),
                  children: [
                    for (var i = 0; i < ids.length; i++)
                      ListTile(
                        key: ValueKey(ids[i]),
                        leading: SizedBox(
                          width: 64,
                          child: FutureBuilder<Uint8List>(
                            future: app.queue.sheet(draft!, i),
                            builder: (_, snap) => snap.hasData
                                ? Image.memory(snap.data!, fit: BoxFit.cover, cacheWidth: 128)
                                : const Icon(Icons.description),
                          ),
                        ),
                        title: Text(app.t('app.sheet_n', {'n': i + 1})),
                        trailing: IconButton(
                          key: Key('remove_$i'),
                          icon: const Icon(Icons.delete_outline),
                          tooltip: app.t('app.remove'),
                          onPressed: _working ? null : () => _run(() => app.queue.removeSheet(draft, i)),
                        ),
                      ),
                  ],
                ),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            ValueChips(label: app.t('app.speed'), values: const [25, 50], value: _speed, unknown: app.t('app.unknown'),
                unit: app.settings.lang == 'uz' ? 'mm/s' : 'мм/с', onChanged: (v) => setState(() => _speed = v)),
            ValueChips(label: app.t('app.gain'), values: const [5, 10, 20], value: _gain, unknown: app.t('app.unknown'),
                unit: app.settings.lang == 'uz' ? 'mm/mV' : 'мм/мВ', onChanged: (v) => setState(() => _gain = v)),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('more_sheet'),
                  icon: const Icon(Icons.add_a_photo),
                  label: Text(app.t('app.more_sheet')),
                  onPressed: _working ? null : _more,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: FilledButton.icon(
                  key: const Key('send_study'),
                  icon: const Icon(Icons.send),
                  label: Text(app.t('app.send_study')),
                  onPressed: _working || ids.isEmpty
                      ? null
                      : () => _run(() async {
                            await app.sendDraft(speed: _speed, gain: _gain);
                            if (context.mounted) Navigator.pop(context);
                          }),
                ),
              ),
            ]),
          ]),
        ),
      ]),
    );
  }
}
