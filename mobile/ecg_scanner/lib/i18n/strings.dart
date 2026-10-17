// RU/UZ strings: assets/i18n/{ru,uz}.json, a copy of configs/scanner/i18n (SD-016).
import 'dart:convert';

import 'package:flutter/services.dart' show rootBundle;

class Strings {
  Strings(this.lang, this._map);

  final String lang;
  final Map<String, String> _map;

  static Future<Strings> load(String lang) async {
    final raw = await rootBundle.loadString('assets/i18n/$lang.json');
    return Strings(lang, (jsonDecode(raw) as Map<String, dynamic>).cast<String, String>());
  }

  /// The text for `key`, `{name}` placeholders filled. A missing key is an error, not a fallback.
  String t(String key, [Map<String, Object> args = const {}]) {
    var s = _map[key];
    if (s == null) throw ArgumentError('missing string $key ($lang)');
    args.forEach((k, v) => s = s!.replaceAll('{$k}', '$v'));
    return s!;
  }
}
