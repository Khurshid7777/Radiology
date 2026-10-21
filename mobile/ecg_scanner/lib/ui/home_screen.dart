import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../upload/upload_queue.dart';
import 'camera_screen.dart';
import 'confirm_screen.dart';
import 'settings_screen.dart';
import 'widgets.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key, required this.app});

  final AppState app;

  String _line(UploadItem item) {
    switch (item.state) {
      case UploadState.queued:
        final s = app.queue.retryIn(item);
        return s == null || s == 0 ? app.t('app.queued') : app.t('app.retry_in', {'seconds': s});
      case UploadState.uploading:
        return app.t('app.uploading');
      case UploadState.failed:
        return app.t('app.error', {'message': item.error ?? ''});
      case UploadState.sent:
        final scan = app.scans[item.scanId];
        if (scan == null) return app.t('app.sent');
        return (scan['messages'] as Map<String, dynamic>)[app.settings.lang]['status'] as String;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: app,
      builder: (context, _) => Scaffold(
        appBar: AppBar(title: Text(app.t('app.title')), actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: app.t('app.settings'),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => SettingsScreen(app: app))),
          ),
        ]),
        body: app.queue.items.isEmpty
            ? Center(child: Padding(padding: const EdgeInsets.all(24), child: Text(app.t('app.empty_history'))))
            : ListView(children: [
                for (final item in app.queue.items)
                  ListTile(
                    leading: Icon(Icons.monitor_heart,
                        color: statusColor(app.scans[item.scanId]?['status'] as String? ?? item.state.name)),
                    title: Text(_line(item)),
                    subtitle: Text(item.createdAt.toLocal().toString().substring(0, 16)),
                    onTap: item.scanId == null
                        ? null
                        : () => Navigator.push(context,
                            MaterialPageRoute(builder: (_) => ConfirmScreen(app: app, scanId: item.scanId!))),
                  ),
              ]),
        floatingActionButton: FloatingActionButton.extended(
          icon: const Icon(Icons.camera_alt),
          label: Text(app.t('app.capture')),
          onPressed: !app.settings.configured
              ? () => Navigator.push(context, MaterialPageRoute(builder: (_) => SettingsScreen(app: app)))
              : () => Navigator.push(context, MaterialPageRoute(builder: (_) => CameraScreen(app: app))),
        ),
      ),
    );
  }
}
