// Photos waiting on the phone are encrypted (TZ addendum §18.9, SD-073): AES-256-GCM
// with a key kept by the platform's keystore (Android Keystore / iOS Keychain through
// flutter_secure_storage), never in the app's files. A sealed photo is
// nonce (12 bytes) | ciphertext | tag (16 bytes).
import 'dart:convert';
import 'dart:typed_data';

import 'package:cryptography/cryptography.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Where the key lives: the platform keystore in the app, memory in tests.
abstract class KeyStore {
  Future<String?> read(String key);
  Future<void> write(String key, String value);
}

class SecureKeyStore implements KeyStore {
  SecureKeyStore([FlutterSecureStorage? storage]) : _s = storage ?? const FlutterSecureStorage();

  final FlutterSecureStorage _s;

  @override
  Future<String?> read(String key) => _s.read(key: key);

  @override
  Future<void> write(String key, String value) => _s.write(key: key, value: value);
}

class MemoryKeyStore implements KeyStore {
  final Map<String, String> values = {};

  @override
  Future<String?> read(String key) async => values[key];

  @override
  Future<void> write(String key, String value) async => values[key] = value;
}

class LocalVault {
  LocalVault(this._store);

  static const _keyName = 'ecgscan.photo_key.v1';
  final KeyStore _store;
  final _aes = AesGcm.with256bits();
  SecretKey? _key;

  Future<SecretKey> _secret() async {
    final cached = _key;
    if (cached != null) return cached;
    final stored = await _store.read(_keyName);
    if (stored != null) return _key = SecretKey(base64Decode(stored));
    final fresh = await _aes.newSecretKey();
    await _store.write(_keyName, base64Encode(await fresh.extractBytes()));
    return _key = fresh;
  }

  Future<Uint8List> seal(Uint8List plain) async {
    final box = await _aes.encrypt(plain, secretKey: await _secret());
    return box.concatenation();
  }

  Future<Uint8List> open(Uint8List sealed) async {
    final box = SecretBox.fromConcatenation(sealed, nonceLength: 12, macLength: 16, copy: false);
    return Uint8List.fromList(await _aes.decrypt(box, secretKey: await _secret()));
  }
}
