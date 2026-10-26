import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'package:local_auth/local_auth.dart';

import 'security/app_lock.dart';
import 'security/local_vault.dart';
import 'state/app_state.dart';
import 'ui/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Photos and settings stay in the app's private folder (never the gallery); waiting
  // photos are encrypted with a key held by the platform keystore (TZ addendum §18.9).
  final app = await AppState.open(await getApplicationSupportDirectory(), keys: SecureKeyStore());
  app.start();
  runApp(EcgScannerApp(app: app));
}

class EcgScannerApp extends StatelessWidget {
  const EcgScannerApp({super.key, required this.app});

  final AppState app;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ECG Scanner',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      builder: (context, child) => AppLock(app: app, auth: LocalAuthentication(), child: child!),
      home: HomeScreen(app: app),
    );
  }
}
