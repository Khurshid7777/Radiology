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
    this.pinHash,
    this.pinSalt,
    this.pinIterations = 120000,
    this.lockAfterMinutes = 5,
    this.biometrics = true,
    this.deviceProfileId,
    this.deviceProfileName,
    this.deviceZones = const [],
    this.zoneMargin = 0.01,
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

  /// App lock (TZ addendum §18.9): PBKDF2 hash and salt of the PIN (base64), never the PIN.
  String? pinHash;
  String? pinSalt;
  int pinIterations;

  /// The lock comes back after this long in the background (0: every time).
  int lockAfterMinutes;

  /// Unlock with the phone's biometrics when it has them (the PIN always works).
  bool biometrics;

  /// The recorder the doctor works with (TZ addendum §18.5, §18.6; S3): its profile id on the
  /// server, its name, and the zones of its sheet with personal data ([x0, y0, x1, y1] in
  /// fractions of the upright sheet), kept here so masking works offline. No profile: the band.
  String? deviceProfileId;
  String? deviceProfileName;
  List<List<double>> deviceZones;
  double zoneMargin;

  bool get hasZones => deviceProfileId != null && deviceZones.isNotEmpty;

  bool get pinSet => pinHash != null && pinSalt != null;

  bool get configured => serverUrl.isNotEmpty && apiKey.isNotEmpty;

  /// A copy with the given fields replaced (the lock fields are kept unless given).
  Settings copyWith({String? serverUrl, String? apiKey, String? lang, bool? maskHeader, bool? autoCapture,
          double? headerBand, int? lockAfterMinutes, bool? biometrics}) =>
      Settings(
        serverUrl: serverUrl ?? this.serverUrl,
        apiKey: apiKey ?? this.apiKey,
        lang: lang ?? this.lang,
        maskHeader: maskHeader ?? this.maskHeader,
        autoCapture: autoCapture ?? this.autoCapture,
        headerBand: headerBand ?? this.headerBand,
        pinHash: pinHash,
        pinSalt: pinSalt,
        pinIterations: pinIterations,
        lockAfterMinutes: lockAfterMinutes ?? this.lockAfterMinutes,
        biometrics: biometrics ?? this.biometrics,
        deviceProfileId: deviceProfileId,
        deviceProfileName: deviceProfileName,
        deviceZones: deviceZones,
        zoneMargin: zoneMargin,
      );

  /// The chosen recorder (null: none; masking falls back to the band).
  void setDevice(Map<String, dynamic>? profile) {
    deviceProfileId = profile?['id'] as String?;
    deviceProfileName = profile == null ? null : '${profile['manufacturer'] ?? ''} ${profile['model'] ?? ''}'.trim();
    deviceZones = [
      for (final z in (profile?['phi_zones'] as List? ?? const []))
        [for (final v in z as List) (v as num).toDouble()]
    ];
    zoneMargin = (profile?['phi_zone_margin'] as num?)?.toDouble() ?? 0.01;
  }

  Map<String, dynamic> toJson() => {
        'server_url': serverUrl,
        'api_key': apiKey,
        'lang': lang,
        'mask_header': maskHeader,
        'auto_capture': autoCapture,
        'header_band': headerBand,
        'pin_hash': pinHash,
        'pin_salt': pinSalt,
        'pin_iterations': pinIterations,
        'lock_after_minutes': lockAfterMinutes,
        'biometrics': biometrics,
        'device_profile_id': deviceProfileId,
        'device_profile_name': deviceProfileName,
        'device_zones': deviceZones,
        'zone_margin': zoneMargin,
      };

  factory Settings.fromJson(Map<String, dynamic> j) => Settings(
        serverUrl: j['server_url'] as String? ?? '',
        apiKey: j['api_key'] as String? ?? '',
        lang: j['lang'] as String? ?? 'ru',
        maskHeader: j['mask_header'] as bool? ?? true,
        autoCapture: j['auto_capture'] as bool? ?? true,
        headerBand: (j['header_band'] as num?)?.toDouble() ?? 0.14,
        pinHash: j['pin_hash'] as String?,
        pinSalt: j['pin_salt'] as String?,
        pinIterations: j['pin_iterations'] as int? ?? 120000,
        lockAfterMinutes: j['lock_after_minutes'] as int? ?? 5,
        biometrics: j['biometrics'] as bool? ?? true,
        deviceProfileId: j['device_profile_id'] as String?,
        deviceProfileName: j['device_profile_name'] as String?,
        deviceZones: [
          for (final z in (j['device_zones'] as List? ?? const []))
            [for (final v in z as List) (v as num).toDouble()]
        ],
        zoneMargin: (j['zone_margin'] as num?)?.toDouble() ?? 0.01,
      );

  static Future<Settings> load(File f) async =>
      await f.exists() ? Settings.fromJson(jsonDecode(await f.readAsString()) as Map<String, dynamic>) : Settings();

  Future<void> save(File f) => f.writeAsString(jsonEncode(toJson()));
}
