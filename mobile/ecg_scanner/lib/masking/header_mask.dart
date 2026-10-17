// Patient header masking before upload (TZ §6.2, §9; SD-008).
//
// v1 masks a fixed top band of the photo: without on-device OCR the name and ID can
// be anywhere in the header, and the TZ says to mask the top band when unsure. The
// band also hides the printed "25 mm/s 10 mm/mV", so the doctor states speed and
// gain on the capture screen (sent as user hints) — otherwise the server asks.
import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// The photo with its top `bandFraction` painted white, re-encoded as JPEG.
/// EXIF is not carried over (the original's metadata never leaves the phone).
Uint8List maskHeader(Uint8List jpeg, {double bandFraction = 0.2, int quality = 95}) {
  final photo = img.decodeImage(jpeg);
  if (photo == null) throw ArgumentError('not a decodable image');
  final upright = img.bakeOrientation(photo);
  final band = (upright.height * bandFraction).round();
  img.fillRect(upright, x1: 0, y1: 0, x2: upright.width - 1, y2: band - 1, color: img.ColorRgb8(255, 255, 255));
  upright.exif = img.ExifData();
  return Uint8List.fromList(img.encodeJpg(upright, quality: quality));
}
