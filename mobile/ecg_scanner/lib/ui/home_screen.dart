import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../state/study_status.dart';
import '../upload/upload_queue.dart';
import 'camera_screen.dart';
import 'confirm_screen.dart';
import 'settings_screen.dart';
import 'sheets_screen.dart';
import 'widgets.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key, required this.app});

  final AppState app;

  String _line(UploadItem item) {
    final scan = app.scans[item.scanId];
    final st = studyStatus(item, scan);
    switch (st) {
      case StudyStatus.draft:
        return app.t('app.st_draft', {'n': item.partIds.length});
      case StudyStatus.queued:
        final s = app.queue.retryIn(item);
        return s == null || s == 0 ? app.t('app.st_queued') : app.t('app.retry_in', {'seconds': s});
      case StudyStatus.failed:
        return app.t('app.st_failed', {'message': item.error ?? ''});
      case StudyStatus.ready:
      case StudyStatus.retake:
        // The server's own words: "ok", "check", or why it must be taken again.
        return '${app.t('app.st_${st.name}')} — ${(scan!['messages'] as Map<String, dynamic>)[app.settings.lang]['status']}';
      case StudyStatus.uploading:
      case StudyStatus.processing:
        return app.t('app.st_${st.name}');
    }
  }

  void _open(BuildContext context, UploadItem item) {
    if (item.state == UploadState.draft) {
      Navigator.push(context, MaterialPageRoute(builder: (_) => SheetsScreen(app: app)));
    } else if (item.scanId != null) {
      Navigator.push(context, MaterialPageRoute(builder: (_) => ConfirmScreen(app: app, scanId: item.scanId!)));
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
                if (app.queue.diskFull)
                  MaterialBanner(
                    key: const Key('disk_full'),
                    content: Text(app.t('app.disk_full')),
                    leading: const Icon(Icons.sd_card_alert, color: Colors.orange),
                    actions: const [SizedBox.shrink()],
                  ),
                for (final item in app.queue.items)
                  ListTile(
                    leading: Icon(Icons.monitor_heart,
                        color: statusColor(app.scans[item.scanId]?['status'] as String? ?? item.state.name)),
                    title: Text(_line(item)),
                    subtitle: Text(item.createdAt.toLocal().toString().substring(0, 16) +
                        (item.partIds.length > 1 ? ' · ${app.t('app.sheets_n', {'n': item.partIds.length})}' : '')),
                    trailing: item.state == UploadState.failed
                        ? IconButton(
                            key: Key('retry_${item.localId}'),
                            icon: const Icon(Icons.refresh),
                            tooltip: app.t('app.retry'),
                            onPressed: () async {
                              await app.queue.retry(item);
                              await app.tick();
                            },
                          )
                        : null,
                    onTap: item.state == UploadState.draft || item.scanId != null ? () => _open(context, item) : null,
                  ),
              ]),
        floatingActionButton: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.end, children: [
          FloatingActionButton.extended(
            key: const Key('several_sheets'),
            heroTag: 'sheets',
            icon: const Icon(Icons.library_add),
            label: Text(app.t('app.several_sheets')),
            onPressed: !app.settings.configured
                ? () => Navigator.push(context, MaterialPageRoute(builder: (_) => SettingsScreen(app: app)))
                : () => Navigator.push(context, MaterialPageRoute(builder: (_) => SheetsScreen(app: app))),
          ),
          const SizedBox(height: 12),
          FloatingActionButton.extended(
            heroTag: 'capture',
            icon: const Icon(Icons.camera_alt),
            label: Text(app.t('app.capture')),
            onPressed: !app.settings.configured
                ? () => Navigator.push(context, MaterialPageRoute(builder: (_) => SettingsScreen(app: app)))
                : () => Navigator.push(context, MaterialPageRoute(builder: (_) => CameraScreen(app: app))),
          ),
        ]),
      ),
    );
  }
}
