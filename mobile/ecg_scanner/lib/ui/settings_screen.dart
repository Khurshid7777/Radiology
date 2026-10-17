import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../state/settings.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.app});

  final AppState app;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _url = TextEditingController(text: widget.app.settings.serverUrl);
  late final TextEditingController _key = TextEditingController(text: widget.app.settings.apiKey);
  late String _lang = widget.app.settings.lang;
  late bool _mask = widget.app.settings.maskHeader;
  late bool _auto = widget.app.settings.autoCapture;

  @override
  Widget build(BuildContext context) {
    final app = widget.app;
    return Scaffold(
      appBar: AppBar(title: Text(app.t('app.settings'))),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        TextField(controller: _url, decoration: InputDecoration(labelText: app.t('app.server_url')),
            keyboardType: TextInputType.url),
        TextField(controller: _key, decoration: InputDecoration(labelText: app.t('app.api_key')), obscureText: true),
        const SizedBox(height: 16),
        Row(children: [
          Text(app.t('app.language')),
          const SizedBox(width: 16),
          SegmentedButton<String>(
            segments: const [ButtonSegment(value: 'ru', label: Text('Русский')),
                             ButtonSegment(value: 'uz', label: Text('Oʻzbekcha'))],
            selected: {_lang},
            onSelectionChanged: (s) => setState(() => _lang = s.first),
          ),
        ]),
        SwitchListTile(
          key: const Key('mask_header'),
          title: Text(app.t('app.mask_header')),
          subtitle: Text(app.t('app.mask_header_note')),
          value: _mask,
          onChanged: (v) => setState(() => _mask = v),
        ),
        SwitchListTile(title: Text(app.t('app.auto_capture')), value: _auto, onChanged: (v) => setState(() => _auto = v)),
        const SizedBox(height: 16),
        FilledButton(
          onPressed: () async {
            await app.saveSettings(Settings(serverUrl: _url.text.trim(), apiKey: _key.text.trim(), lang: _lang,
                maskHeader: _mask, autoCapture: _auto, headerBand: app.settings.headerBand));
            app.start();
            if (context.mounted) Navigator.pop(context);
          },
          child: Text(app.t('app.save')),
        ),
      ]),
    );
  }
}
