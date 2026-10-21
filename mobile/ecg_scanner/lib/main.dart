import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'state/app_state.dart';
import 'ui/home_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final app = await AppState.open(await getApplicationSupportDirectory());
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
      home: HomeScreen(app: app),
    );
  }
}
