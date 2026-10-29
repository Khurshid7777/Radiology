// The /scans API (TZ §4; SD-040): upload, status, confirmation, analysis, frame-check config.
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

class ApiError implements Exception {
  ApiError(this.status, this.message);

  final int status;
  final String message;

  /// Worth retrying later: no network, server overload or restart.
  bool get transient => status == 0 || status == 408 || status == 429 || status >= 500;

  @override
  String toString() => 'ApiError($status): $message';
}

class ScanHints {
  const ScanHints(
      {this.paperSpeedMmS,
      this.gainMmMv,
      this.headerMasked = false,
      this.capture,
      this.device,
      this.pages = false,
      this.deviceProfile});

  final double? paperSpeedMmS;
  final double? gainMmMv;
  final bool headerMasked;

  /// How the photo was taken (TZ addendum §18.13): ecgscan.contract.Capture as JSON.
  final Map<String, dynamic>? capture;

  /// The phone model, no personal data.
  final String? device;

  /// Several sheets of one study (TZ addendum §18.4): the server stitches them in order.
  final bool pages;

  /// The recorder's profile id on the server (`configs/scanner/devices/<id>.yaml`; S2, S3).
  final String? deviceProfile;

  Map<String, String> toFields() => {
        'client': 'flutter',
        'header_masked': '$headerMasked',
        if (paperSpeedMmS != null) 'paper_speed_mm_s': '$paperSpeedMmS',
        if (gainMmMv != null) 'gain_mm_mv': '$gainMmMv',
        if (capture != null) 'capture': jsonEncode(capture),
        'device': ?device,
        if (pages) 'pages': 'true',
        'device_profile': ?deviceProfile,
      };

  Map<String, dynamic> toJson() => {
        'paper_speed_mm_s': paperSpeedMmS,
        'gain_mm_mv': gainMmMv,
        'header_masked': headerMasked,
        'capture': capture,
        'device': device,
        'pages': pages,
        'device_profile': deviceProfile,
      };

  factory ScanHints.fromJson(Map<String, dynamic> j) => ScanHints(
        paperSpeedMmS: (j['paper_speed_mm_s'] as num?)?.toDouble(),
        gainMmMv: (j['gain_mm_mv'] as num?)?.toDouble(),
        headerMasked: j['header_masked'] as bool? ?? false,
        capture: j['capture'] as Map<String, dynamic>?,
        device: j['device'] as String?,
        pages: j['pages'] as bool? ?? false,
        deviceProfile: j['device_profile'] as String?,
      );
}

class ScanApi {
  ScanApi({required this.baseUrl, required this.apiKey, http.Client? client}) : _http = client ?? http.Client();

  final String baseUrl;
  final String apiKey;
  final http.Client _http;

  Uri _u(String path) => Uri.parse('${baseUrl.replaceAll(RegExp(r'/+$'), '')}$path');
  Map<String, String> get _auth => {'X-API-Key': apiKey};

  Future<Map<String, dynamic>> _json(Future<http.BaseResponse> Function() call) async {
    final http.Response r;
    try {
      final resp = await call();
      r = resp is http.StreamedResponse ? await http.Response.fromStream(resp) : resp as http.Response;
    } on Exception catch (e) {
      throw ApiError(0, e.toString());
    }
    if (r.statusCode >= 300) {
      String msg = r.body;
      try {
        msg = (jsonDecode(r.body) as Map<String, dynamic>)['detail'].toString();
      } on Object catch (_) {}
      throw ApiError(r.statusCode, msg);
    }
    return jsonDecode(utf8.decode(r.bodyBytes)) as Map<String, dynamic>;
  }

  /// POST /scans: the scan id. `more`: the next sheets of the study, in order (`file` is the first).
  Future<String> submit(Uint8List jpeg, ScanHints hints, {List<Uint8List> more = const []}) async {
    final body = await _json(() {
      final req = http.MultipartRequest('POST', _u('/scans'))
        ..headers.addAll(_auth)
        ..fields.addAll(hints.toFields())
        ..files.add(http.MultipartFile.fromBytes('file', jpeg, filename: 'ecg.jpg'));
      for (var i = 0; i < more.length; i++) {
        req.files.add(http.MultipartFile.fromBytes('parts', more[i], filename: 'ecg_${i + 2}.jpg'));
      }
      return _http.send(req);
    });
    return body['scan_id'] as String;
  }

  Future<Map<String, dynamic>> get(String scanId) => _json(() => _http.get(_u('/scans/$scanId'), headers: _auth));

  Future<Map<String, dynamic>> confirm(String scanId, String decision,
          {double? paperSpeedMmS, double? gainMmMv, String? layout, String? comment}) =>
      _json(() => _http.post(_u('/scans/$scanId/confirm'),
          headers: {..._auth, 'Content-Type': 'application/json'},
          body: jsonEncode({
            'decision': decision,
            'paper_speed_mm_s': ?paperSpeedMmS,
            'gain_mm_mv': ?gainMmMv,
            'layout': ?layout,
            'comment': ?comment,
          })));

  Future<Map<String, dynamic>> analyze(String scanId) =>
      _json(() => _http.post(_u('/scans/$scanId/analysis'), headers: {..._auth, 'Content-Type': 'application/json'},
          body: '{}'));

  /// GET /scans/config/devices: the recorder profiles ({id, manufacturer, model, layouts,
  /// phi_zones, phi_zone_margin}), for masking the right zones before upload (S3).
  Future<List<Map<String, dynamic>>> devices() async {
    final j = await _json(() => _http.get(_u('/scans/config/devices'), headers: _auth));
    return [for (final d in (j['devices'] as List? ?? const [])) (d as Map).cast<String, dynamic>()];
  }

  /// GET /scans/config/features: which P2 features (consult, archive, qr, signature) the
  /// server has on (TZ 18.11, SD-097). The app shows a feature only when it is true; today
  /// all are off. A server without the endpoint means all off.
  Future<Map<String, bool>> features() async {
    final Map<String, dynamic> j;
    try {
      j = await _json(() => _http.get(_u('/scans/config/features'), headers: _auth));
    } on ApiError catch (e) {
      if (e.status == 404) return const {};
      rethrow;
    }
    return {for (final e in ((j['features'] as Map?) ?? const {}).entries) e.key as String: e.value == true};
  }

  /// GET /scans/config/capture-qc: {config, hints: {ru, uz}}.
  Future<Map<String, dynamic>> captureQcConfig() =>
      _json(() => _http.get(_u('/scans/config/capture-qc'), headers: _auth));

  /// The trace overlay preview (JPEG) for the confirmation screen.
  Future<Uint8List> overlay(String scanId) async {
    final r = await _http.get(_u('/scans/$scanId/artifacts/trace_overlay_preview'), headers: _auth);
    if (r.statusCode != 200) throw ApiError(r.statusCode, 'overlay unavailable');
    return r.bodyBytes;
  }
}
