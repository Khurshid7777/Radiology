import 'package:flutter/material.dart';

import '../state/app_state.dart';

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
  late bool _bio = widget.app.settings.biometrics;
  late int _lockAfter = widget.app.settings.lockAfterMinutes;

  /// Recorder profiles from the server (S3); null while loading, empty when offline or none.
  List<Map<String, dynamic>>? _devices;
  bool _devicesFailed = false;
  late String? _deviceId = widget.app.settings.deviceProfileId;

  @override
  void initState() {
    super.initState();
    _loadDevices();
  }

  Future<void> _loadDevices() async {
    if (!widget.app.settings.configured) {
      setState(() => _devices = const []);
      return;
    }
    try {
      final d = await widget.app.api.devices();
      if (mounted) setState(() => _devices = d);
    } on Object catch (_) {
      if (mounted) {
        setState(() {
          _devices = const [];
          _devicesFailed = true;
        });
      }
    }
  }

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
        ListTile(
          key: const Key('device'),
          title: Text(app.t('app.device')),
          subtitle: _devicesFailed ? Text(app.t('app.devices_unavailable')) : null,
          trailing: _devices == null
              ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
              : DropdownButton<String?>(
                  value: _deviceItems(app).any((i) => i.value == _deviceId) ? _deviceId : null,
                  items: _deviceItems(app),
                  onChanged: (v) => setState(() => _deviceId = v),
                ),
        ),
        SwitchListTile(title: Text(app.t('app.auto_capture')), value: _auto, onChanged: (v) => setState(() => _auto = v)),
        SwitchListTile(title: Text(app.t('app.biometrics')), value: _bio, onChanged: (v) => setState(() => _bio = v)),
        ListTile(
          title: Text(app.t('app.lock_after')),
          trailing: DropdownButton<int>(
            value: _lockAfter,
            items: [for (final m in const [0, 1, 5, 15]) DropdownMenuItem(value: m, child: Text('$m'))],
            onChanged: (v) => setState(() => _lockAfter = v ?? _lockAfter),
          ),
        ),
        const SizedBox(height: 16),
        FilledButton(
          onPressed: () async {
            final next = app.settings.copyWith(serverUrl: _url.text.trim(), apiKey: _key.text.trim(),
                lang: _lang, maskHeader: _mask, autoCapture: _auto, biometrics: _bio, lockAfterMinutes: _lockAfter);
            if (_deviceId != app.settings.deviceProfileId) {
              final profile = _devices?.where((d) => d['id'] == _deviceId).firstOrNull;
              if (profile != null || _deviceId == null) next.setDevice(profile);
            }
            await app.saveSettings(next);
            app.start();
            if (context.mounted) Navigator.pop(context);
          },
          child: Text(app.t('app.save')),
        ),
      ]),
    );
  }

  /// "None" first, then the server's profiles; offline, the stored choice stays listed.
  List<DropdownMenuItem<String?>> _deviceItems(AppState app) {
    final s = app.settings;
    final listed = [
      for (final d in _devices ?? const <Map<String, dynamic>>[])
        DropdownMenuItem<String?>(
            value: d['id'] as String, child: Text('${d['manufacturer'] ?? ''} ${d['model'] ?? ''}'.trim())),
    ];
    return [
      DropdownMenuItem<String?>(value: null, child: Text(app.t('app.device_none'))),
      ...listed,
      if (s.deviceProfileId != null && !listed.any((i) => i.value == s.deviceProfileId))
        DropdownMenuItem<String?>(value: s.deviceProfileId, child: Text(s.deviceProfileName ?? s.deviceProfileId!)),
    ];
  }
}
