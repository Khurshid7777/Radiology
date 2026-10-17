// App settings, kept in the app's private folder.
//
// The API key is stored there in plain form: the folder is sandboxed per app; a
// keystore-backed store (flutter_secure_storage) is the next step if the threat
// model asks for it (SD-047).
import 'dart:convert';
import 'dart:io';

class Settings {
  Settings({
    this.serverUrl = '',
    this.apiKey = '',
    this.lang = 'ru',
    this.maskHeader = true,
    this.autoCapture = true,
    this.headerBand = 0.14,
  });

  String serverUrl;
  String apiKey;
  String lang;

  /// TZ §6.2: on by default.
  bool maskHeader;
  bool autoCapture;

  /// Default share of the photo height masked from the top (TZ §6.2: mask the top band
  /// when unsure). Synthetic pages: name/ID end above 8.5 % of the page, the first trace
  /// row starts below 18 % (a 20 % band cut a row in 1 of 8 photos). Adjusted per photo.
  double headerBand;

  bool get configured => serverUrl.isNotEmpty && apiKey.isNotEmpty;

  Map<String, dynamic> toJson() => {
        'server_url': serverUrl,
        'api_key': apiKey,
        'lang': lang,
        'mask_header': maskHeader,
        'auto_capture': autoCapture,
        'header_band': headerBand,
      };

  factory Settings.fromJson(Map<String, dynamic> j) => Settings(
        serverUrl: j['server_url'] as String? ?? '',
        apiKey: j['api_key'] as String? ?? '',
        lang: j['lang'] as String? ?? 'ru',
        maskHeader: j['mask_header'] as bool? ?? true,
        autoCapture: j['auto_capture'] as bool? ?? true,
        headerBand: (j['header_band'] as num?)?.toDouble() ?? 0.14,
      );

  static Future<Settings> load(File f) async =>
      await f.exists() ? Settings.fromJson(jsonDecode(await f.readAsString()) as Map<String, dynamic>) : Settings();

  Future<void> save(File f) => f.writeAsString(jsonEncode(toJson()));
}
