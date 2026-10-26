// App lock (TZ addendum §18.9, SD-073): a PIN set on first start, biometrics as a
// shortcut when the phone has them, and the lock comes back after the app has been in
// the background longer than `lockAfterMinutes`. While the app is not in front its
// screen is covered, so the app switcher's snapshot shows no patient data (iOS; on
// Android FLAG_SECURE in MainActivity blocks screenshots and the snapshot).
//
// The PIN is never stored: PBKDF2-HMAC-SHA256 of it with a random salt is.
import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:cryptography/cryptography.dart';
import 'package:flutter/material.dart';
import 'package:local_auth/local_auth.dart';

import '../state/app_state.dart';

const pinIterations = 120000;
const minPinLength = 4;
const maxPinLength = 8;

/// Wrong PINs in a row before the entry pauses, and the pause.
const maxAttempts = 5;
const attemptPause = Duration(seconds: 30);

Future<String> hashPin(String pin, List<int> salt, {int iterations = pinIterations}) async {
  final kdf = Pbkdf2(macAlgorithm: Hmac.sha256(), iterations: iterations, bits: 256);
  final key = await kdf.deriveKey(secretKey: SecretKey(utf8.encode(pin)), nonce: salt);
  return base64Encode(await key.extractBytes());
}

List<int> newSalt() {
  final r = math.Random.secure();
  return List<int>.generate(16, (_) => r.nextInt(256));
}

bool validPin(String pin) => RegExp('^[0-9]{$minPinLength,$maxPinLength}\$').hasMatch(pin);

/// Constant-time comparison of two hashes.
bool sameHash(String a, String b) {
  if (a.length != b.length) return false;
  var d = 0;
  for (var i = 0; i < a.length; i++) {
    d |= a.codeUnitAt(i) ^ b.codeUnitAt(i);
  }
  return d == 0;
}

/// When the lock comes back: after `lockAfter` in the background (zero: on every return).
bool shouldLock(DateTime? wentAway, DateTime now, Duration lockAfter) =>
    wentAway != null && now.difference(wentAway) >= lockAfter;

class AppLock extends StatefulWidget {
  const AppLock({super.key, required this.app, required this.child, this.iterations = pinIterations, this.auth});

  final AppState app;
  final Widget child;
  final int iterations;

  /// The platform's biometrics (tests pass none: PIN only).
  final LocalAuthentication? auth;

  @override
  State<AppLock> createState() => _AppLockState();
}

class _AppLockState extends State<AppLock> with WidgetsBindingObserver {
  bool _locked = true;
  bool _covered = false;
  DateTime? _wentAway;

  AppState get app => widget.app;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.inactive:
      case AppLifecycleState.hidden:
        setState(() => _covered = true);
      case AppLifecycleState.paused:
        _wentAway ??= DateTime.now();
        setState(() => _covered = true);
      case AppLifecycleState.resumed:
        final lock = shouldLock(_wentAway, DateTime.now(), Duration(minutes: app.settings.lockAfterMinutes));
        _wentAway = null;
        setState(() {
          _covered = false;
          if (lock) _locked = true;
        });
      case AppLifecycleState.detached:
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    final Widget body;
    if (!app.settings.pinSet) {
      body = PinScreen(app: app, iterations: widget.iterations, setting: true, onDone: () => setState(() => _locked = false));
    } else if (_locked) {
      body = PinScreen(app: app, iterations: widget.iterations, auth: widget.auth, onDone: () => setState(() => _locked = false));
    } else {
      body = widget.child;
    }
    return Stack(children: [
      body,
      if (_covered) const Positioned.fill(child: ColoredBox(color: Colors.white, child: Center(child: Icon(Icons.lock, size: 64)))),
    ]);
  }
}

class PinScreen extends StatefulWidget {
  const PinScreen(
      {super.key, required this.app, required this.onDone, this.setting = false, this.iterations = pinIterations, this.auth});

  final AppState app;
  final VoidCallback onDone;

  /// First start: the PIN is chosen (entered twice) instead of checked.
  final bool setting;
  final int iterations;
  final LocalAuthentication? auth;

  @override
  State<PinScreen> createState() => _PinScreenState();
}

class _PinScreenState extends State<PinScreen> {
  final _pin = TextEditingController();
  String? _first;
  String? _error;
  int _wrong = 0;
  DateTime? _pausedUntil;
  bool _working = false;

  AppState get app => widget.app;

  @override
  void initState() {
    super.initState();
    if (!widget.setting && app.settings.biometrics) unawaited(_biometric());
  }

  Future<void> _biometric() async {
    final auth = widget.auth;
    if (auth == null) return;
    try {
      if (!await auth.canCheckBiometrics) return;
      final ok = await auth.authenticate(localizedReason: app.t('app.unlock_reason'), biometricOnly: true);
      if (ok && mounted) widget.onDone();
    } on Object catch (_) {
      // No biometrics enrolled or the sensor refused: the PIN stays.
    }
  }

  Future<void> _submit() async {
    final pin = _pin.text;
    _pin.clear();
    if (widget.setting) {
      if (!validPin(pin)) return setState(() => _error = app.t('app.pin_rule'));
      if (_first == null) {
        return setState(() {
          _first = pin;
          _error = null;
        });
      }
      if (_first != pin) {
        return setState(() {
          _first = null;
          _error = app.t('app.pin_mismatch');
        });
      }
      setState(() => _working = true);
      final salt = newSalt();
      final s = app.settings
        ..pinSalt = base64Encode(salt)
        ..pinHash = await hashPin(pin, salt, iterations: widget.iterations)
        ..pinIterations = widget.iterations;
      await app.saveSettings(s);
      widget.onDone();
      return;
    }
    final until = _pausedUntil;
    if (until != null && DateTime.now().isBefore(until)) {
      return setState(() => _error = app.t('app.pin_wait', {'seconds': until.difference(DateTime.now()).inSeconds + 1}));
    }
    setState(() => _working = true);
    final h = await hashPin(pin, base64Decode(app.settings.pinSalt!), iterations: app.settings.pinIterations);
    if (!mounted) return;
    if (sameHash(h, app.settings.pinHash!)) {
      widget.onDone();
      return;
    }
    setState(() {
      _working = false;
      _wrong++;
      _error = app.t('app.pin_wrong');
      if (_wrong >= maxAttempts) {
        _wrong = 0;
        _pausedUntil = DateTime.now().add(attemptPause);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final title = widget.setting ? app.t(_first == null ? 'app.pin_set' : 'app.pin_repeat') : app.t('app.pin_enter');
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            const Icon(Icons.lock_outline, size: 48),
            const SizedBox(height: 16),
            Text(title, style: Theme.of(context).textTheme.titleMedium, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            TextField(
              key: const Key('pin'),
              controller: _pin,
              obscureText: true,
              keyboardType: TextInputType.number,
              maxLength: maxPinLength,
              textAlign: TextAlign.center,
              autofocus: true,
              onSubmitted: (_) => _working ? null : _submit(),
            ),
            if (_error != null) Text(_error!, key: const Key('pin_error'), style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 8),
            FilledButton(key: const Key('pin_ok'), onPressed: _working ? null : _submit, child: Text(app.t('app.done'))),
            if (!widget.setting && app.settings.biometrics && widget.auth != null)
              TextButton.icon(
                  icon: const Icon(Icons.fingerprint), label: Text(app.t('app.biometrics')), onPressed: _biometric),
          ]),
        ),
      ),
    );
  }
}
