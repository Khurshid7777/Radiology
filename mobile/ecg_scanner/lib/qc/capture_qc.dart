// Frame quality checks of the camera screen (TZ §6.1, SD-014, SD-046).
//
// A line-by-line port of `ecgscan.capture_qc` (Python, the server side): the
// algorithm is specified in that module's docstring, and both implementations are
// tested on the same reference frames (tests/scanner/fixtures/capture_qc). Keep the
// two in step: any change here needs the same change there, and the reverse.

import 'dart:math' as math;
import 'dart:typed_data';

const hintOrder = ['closer', 'glare', 'steady', 'level', 'light', 'bright', 'shadow', 'frame'];

class FrameQc {
  FrameQc({
    required this.width,
    required this.height,
    required this.factor,
    required this.brightness,
    required this.glare,
    required this.sharpness,
    required this.tiltDeg,
    required this.periodPx,
    required this.pxPerMm,
    required this.sheet,
    required this.hints,
    this.shadow,
    this.corners,
    this.sheetWhole,
    this.sheetArea = 0.0,
    this.sheetBrightness,
  });

  final int width;
  final int height;
  final int factor;
  final double brightness;
  final double glare;
  final double sharpness;
  final double? tiltDeg;
  final double? periodPx; // 5-mm period on the working image
  final double? pxPerMm; // on the captured photo
  final double sheet;
  final List<String> hints;
  final double? shadow; // brightness spread over 4 x 4 zones of the sheet
  final List<List<double>>? corners; // sheet TL, TR, BR, BL in captured-photo px
  final bool? sheetWhole; // the sheet does not run along an edge of the frame
  final double sheetArea; // share of the frame the sheet covers (step 9)
  final double? sheetBrightness; // mean grey of the sheet (step 10)

  bool get ok => hints.isEmpty;
}

/// A grey image as rows of doubles (row-major, `w` columns).
class Plane {
  Plane(this.w, this.h) : data = Float64List(w * h);
  final int w;
  final int h;
  final Float64List data;

  double at(int y, int x) => data[y * w + x];
}

/// Grey and chroma of an RGB frame (`rgb` = w*h*3 bytes, the order a camera gives after YUV conversion).
(Plane, Plane) channels(Uint8List rgb, int w, int h) {
  final g = Plane(w, h), c = Plane(w, h);
  for (var i = 0, p = 0; i < w * h; i++, p += 3) {
    final r = rgb[p].toDouble(), gg = rgb[p + 1].toDouble(), b = rgb[p + 2].toDouble();
    g.data[i] = 0.299 * r + 0.587 * gg + 0.114 * b;
    c.data[i] = math.max(r, math.max(gg, b)) - math.min(r, math.min(gg, b));
  }
  return (g, c);
}

(Plane, int) reduce(Plane g, int workWidth) {
  final k = math.max(1, (g.w / workWidth).ceil());
  final w = g.w ~/ k, h = g.h ~/ k;
  final out = Plane(w, h);
  if (k == 1) {
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        out.data[y * w + x] = g.data[y * g.w + x];
      }
    }
    return (out, 1);
  }
  final n = (k * k).toDouble();
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var s = 0.0;
      for (var dy = 0; dy < k; dy++) {
        final row = (y * k + dy) * g.w + x * k;
        for (var dx = 0; dx < k; dx++) {
          s += g.data[row + dx];
        }
      }
      out.data[y * w + x] = s / n;
    }
  }
  return (out, k);
}

double _mean(Plane p) {
  var s = 0.0;
  for (final v in p.data) {
    s += v;
  }
  return s / p.data.length;
}

double sharpness(Plane w) {
  final vals = <double>[];
  for (var y = 1; y < w.h - 1; y++) {
    for (var x = 1; x < w.w - 1; x++) {
      vals.add(-4.0 * w.at(y, x) + w.at(y - 1, x) + w.at(y + 1, x) + w.at(y, x - 1) + w.at(y, x + 1));
    }
  }
  final m = vals.fold<double>(0, (a, b) => a + b) / vals.length;
  return vals.fold<double>(0, (a, b) => a + (b - m) * (b - m)) / vals.length;
}

double glareShare(Plane w, double level) {
  var hits = 0, total = 0;
  for (var y = 1; y < w.h - 1; y++) {
    for (var x = 1; x < w.w - 1; x++) {
      var s = 0.0;
      for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
          s += w.at(y + dy, x + dx);
        }
      }
      total++;
      if (s / 9.0 >= level) hits++;
    }
  }
  return total == 0 ? 0.0 : hits / total;
}

Float64List movingAverage(Float64List x, int win) {
  if (win <= 1) return x;
  final n = x.length, pad = win ~/ 2;
  final c = Float64List(n + win);
  for (var i = 0; i < n + win - 1; i++) {
    final j = math.min(n - 1, math.max(0, i - pad));
    c[i + 1] = c[i] + x[j];
  }
  final out = Float64List(n);
  for (var i = 0; i < n; i++) {
    out[i] = (c[i + win] - c[i]) / win;
  }
  return out;
}

List<Float64List> bandProfiles(Plane q, {required int band, required int detrend, required int smooth}) {
  final rows = <Float64List>[];
  for (var y = 0; y + band <= q.h; y += band) {
    final p = Float64List(q.w);
    for (var x = 0; x < q.w; x++) {
      var s = 0.0;
      for (var r = y; r < y + band; r++) {
        s += q.at(r, x);
      }
      p[x] = s / band;
    }
    final trend = movingAverage(p, detrend);
    final d = Float64List(q.w);
    for (var x = 0; x < q.w; x++) {
      d[x] = p[x] - trend[x];
    }
    rows.add(movingAverage(d, smooth));
  }
  return rows;
}

double _parabola(double y0, double y1, double y2) {
  final den = y0 - 2.0 * y1 + y2;
  if (den >= 0) return 0.0;
  return math.min(0.5, math.max(-0.5, 0.5 * (y0 - y2) / den));
}

int _argmax(List<double> v) {
  var k = 0;
  for (var i = 1; i < v.length; i++) {
    if (v[i] > v[k]) k = i;
  }
  return k;
}

double? gridPeriod(List<Float64List> profiles,
    {required int minLag, required int harmonics, required double combTol}) {
  if (profiles.isEmpty) return null;
  final n = profiles.first.length;
  final top = (0.4 * n).floor();
  if (top ~/ 2 <= minLag + 1) return null;
  final a = List<double>.filled(top + 2, 0.0);
  for (var lag = 0; lag < top + 2; lag++) {
    var s = 0.0;
    for (final p in profiles) {
      for (var x = 0; x < n - lag; x++) {
        s += p[x] * p[x + lag];
      }
    }
    a[lag] = s / n;
  }
  if (a[0] <= 0) return null;
  final lags = <int>[], scores = <double>[];
  for (var lag = minLag; lag <= top ~/ 2; lag++) {
    final hMax = math.min(harmonics, top ~/ lag);
    var s = 0.0;
    for (var h = 1; h <= hMax; h++) {
      final lo = h * lag - h ~/ 2, hi = math.min(a.length, h * lag + h ~/ 2 + 1);
      var m = a[lo];
      for (var i = lo + 1; i < hi; i++) {
        if (a[i] > m) m = a[i];
      }
      s += m;
    }
    lags.add(lag);
    scores.add(s / hMax);
  }
  final best = scores.reduce(math.max);
  var m = lags[scores.indexWhere((s) => s >= combTol * best)];
  m = m - 1 + _argmax([a[m - 1], a[m], a[m + 1]]);
  if (a[m] <= 0.05 * a[0]) return null; // no periodic structure: no grid
  return m + _parabola(a[m - 1], a[m], a[m + 1]);
}

double _median(List<double> v) {
  final s = [...v]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2.0;
}

double? tiltDeg(List<Float64List> profiles, double period, {required int band}) {
  final j = math.max(1, (1.5 * period / band).floor());
  final sMax = math.max(1, (period / 2).floor() - 1);
  final n = profiles.isEmpty ? 0 : profiles.first.length;
  final shifts = <double>[];
  for (var i = 0; i < profiles.length - j; i++) {
    final p = profiles[i], r = profiles[i + j];
    final xc = <double>[];
    for (var s = -sMax - 1; s <= sMax + 1; s++) {
      final p0 = math.max(0, -s), r0 = math.max(0, s), len = n - s.abs();
      var acc = 0.0;
      for (var x = 0; x < len; x++) {
        acc += p[p0 + x] * r[r0 + x];
      }
      xc.add(acc / n);
    }
    final k = 1 + _argmax(xc.sublist(1, xc.length - 1));
    if (xc[k] > 0) shifts.add(k - sMax - 1 + _parabola(xc[k - 1], xc[k], xc[k + 1]));
  }
  if (shifts.isEmpty) return null;
  return math.atan(_median(shifts) / (j * band)) * 180.0 / math.pi;
}

/// Python's round(): halves to even.
int _roundHalfEven(double x) {
  final f = x.floorToDouble(), d = x - f;
  if (d > 0.5) return f.toInt() + 1;
  if (d < 0.5) return f.toInt();
  return f.toInt().isEven ? f.toInt() : f.toInt() + 1;
}

double _shiftDiff(Plane q, int bx0, int by0, int block, int dx, int dy) {
  final y0 = math.max(0, -dy), y1 = math.min(block, block - dy);
  var s = 0.0;
  var cnt = 0;
  for (var y = y0; y < y1; y++) {
    for (var x = 0; x < block - dx; x++) {
      s += (q.at(by0 + y, bx0 + x) - q.at(by0 + y + dy, bx0 + x + dx)).abs();
      cnt++;
    }
  }
  return cnt == 0 ? 0.0 : s / cnt;
}

/// Step 8: paper flag per block (rows of blocks) and the block size in working px.
(List<List<bool>>, int) paperBlocks(Plane q, double? period, double? tilt, {required int block, required double ratio}) {
  if (period == null) return (<List<bool>>[], block);
  final big = math.max(2, _roundHalfEven(period));
  final half = math.max(1, big ~/ 2);
  final tan = math.tan((tilt ?? 0.0) * math.pi / 180.0);
  final dBig = -_roundHalfEven(big * tan), dHalf = -_roundHalfEven(half * tan);
  final b = math.max(block, 2 * (big + dBig.abs()));
  final hb = q.h ~/ b, wb = q.w ~/ b;
  if (wb == 0) return (<List<bool>>[], b);
  final rows = <List<bool>>[];
  for (var by = 0; by < hb; by++) {
    final row = <bool>[];
    for (var bx = 0; bx < wb; bx++) {
      row.add(_shiftDiff(q, bx * b, by * b, b, half, dHalf) > ratio * _shiftDiff(q, bx * b, by * b, b, big, dBig) + 1e-9);
    }
    rows.add(row);
  }
  return (rows, b);
}

double sheetFraction(Plane q, double? period, double? tilt, {required int block, required double ratio}) {
  final (rows, _) = paperBlocks(q, period, tilt, block: block, ratio: ratio);
  var n = 0, paper = 0;
  for (final r in rows) {
    n += r.length;
    paper += r.where((v) => v).length;
  }
  return n == 0 ? 0.0 : paper / n;
}

const _nb = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Step 9: paper blocks OR their closing (dilation, then erosion with outside the grid unset).
List<List<bool>> closeBlocks(List<List<bool>> rows) {
  final hb = rows.length, wb = hb == 0 ? 0 : rows[0].length;
  bool inGrid(int y, int x) => y >= 0 && y < hb && x >= 0 && x < wb;
  final dil = List.generate(
      hb, (y) => List.generate(wb, (x) => rows[y][x] || _nb.any((d) => inGrid(y + d.$1, x + d.$2) && rows[y + d.$1][x + d.$2])));
  return List.generate(
      hb,
      (y) => List.generate(
          wb, (x) => rows[y][x] || (dil[y][x] && _nb.every((d) => inGrid(y + d.$1, x + d.$2) && dil[y + d.$1][x + d.$2]))));
}

/// Step 9: the largest 4-connected group of set blocks, as (by, bx) sorted row-major.
List<(int, int)> largestGroup(List<List<bool>> rows) {
  final hb = rows.length, wb = hb == 0 ? 0 : rows[0].length;
  final seen = List.generate(hb, (_) => List<bool>.filled(wb, false));
  var best = <(int, int)>[];
  for (var y = 0; y < hb; y++) {
    for (var x = 0; x < wb; x++) {
      if (!rows[y][x] || seen[y][x]) continue;
      final group = <(int, int)>[];
      final stack = <(int, int)>[(y, x)];
      seen[y][x] = true;
      while (stack.isNotEmpty) {
        final (cy, cx) = stack.removeLast();
        group.add((cy, cx));
        for (final (ny, nx) in [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]) {
          if (ny >= 0 && ny < hb && nx >= 0 && nx < wb && rows[ny][nx] && !seen[ny][nx]) {
            seen[ny][nx] = true;
            stack.add((ny, nx));
          }
        }
      }
      if (group.length > best.length) best = group;
    }
  }
  best.sort((a, b) => a.$1 != b.$1 ? a.$1.compareTo(b.$1) : a.$2.compareTo(b.$2));
  return best;
}

/// Step 9: TL, TR, BR, BL of the group's block corners (working px).
List<List<double>>? sheetCorners(List<(int, int)> group, int block) {
  if (group.isEmpty) return null;
  final keys = <double Function(int, int)>[
    (x, y) => -(x + y).toDouble(),
    (x, y) => (x - y).toDouble(),
    (x, y) => (x + y).toDouble(),
    (x, y) => (y - x).toDouble(),
  ];
  final best = List<List<double>?>.filled(4, null);
  final score = List<double>.filled(4, double.negativeInfinity);
  for (final (by, bx) in group) {
    for (final (x, y) in [
      (bx * block, by * block),
      ((bx + 1) * block, by * block),
      ((bx + 1) * block, (by + 1) * block),
      (bx * block, (by + 1) * block),
    ]) {
      for (var i = 0; i < 4; i++) {
        final v = keys[i](x, y);
        if (v > score[i]) {
          score[i] = v;
          best[i] = [x.toDouble(), y.toDouble()];
        }
      }
    }
  }
  return best.cast<List<double>>();
}

double _blockMean(Plane w, int by, int bx, int block) {
  var s = 0.0;
  for (var y = by * block; y < (by + 1) * block; y++) {
    for (var x = bx * block; x < (bx + 1) * block; x++) {
      s += w.at(y, x);
    }
  }
  return s / (block * block);
}

/// Step 10: the mean of the sheet's block levels.
double? sheetBrightness(Plane w, List<(int, int)> group, int block) {
  if (group.isEmpty) return null;
  var s = 0.0;
  for (final (by, bx) in group) {
    s += _blockMean(w, by, bx, block);
  }
  return s / group.length;
}

/// Step 10: (max - min) / max of the zone levels over a 4 x 4 split of the sheet.
double? shadowSpread(Plane w, List<(int, int)> group, int block) {
  if (group.isEmpty) return null;
  final members = group.toSet();
  final inner = group
      .where((b) =>
          members.contains((b.$1 - 1, b.$2)) &&
          members.contains((b.$1 + 1, b.$2)) &&
          members.contains((b.$1, b.$2 - 1)) &&
          members.contains((b.$1, b.$2 + 1)))
      .toList();
  final use = inner.isEmpty ? group : inner;
  var y0 = use.first.$1, y1 = y0, x0 = use.first.$2, x1 = x0;
  for (final (by, bx) in use) {
    y0 = math.min(y0, by);
    y1 = math.max(y1, by);
    x0 = math.min(x0, bx);
    x1 = math.max(x1, bx);
  }
  final ny = y1 - y0 + 1, nx = x1 - x0 + 1;
  final zones = <(int, int), List<double>>{};
  for (final (by, bx) in use) {
    zones.putIfAbsent((4 * (by - y0) ~/ ny, 4 * (bx - x0) ~/ nx), () => []).add(_blockMean(w, by, bx, block));
  }
  if (zones.length < 2) return null;
  final levels = zones.values.map((v) {
    final sv = [...v]..sort();
    return sv[(0.75 * (sv.length - 1)).floor()];
  }).toList();
  final top = levels.reduce(math.max);
  return top <= 0 ? 0.0 : (top - levels.reduce(math.min)) / top;
}

/// Python's round(x, 1).
double _round1(double x) => _roundHalfEven(x * 10) / 10.0;

/// QC of one frame. `scaleToCapture` = captured photo width / this frame's width.
/// `config` is the `config` object of GET /scans/config/capture-qc.
FrameQc checkFrame(Uint8List rgb, int width, int height, Map<String, dynamic> config,
    {double scaleToCapture = 1.0}) {
  final a = config['algorithm'] as Map<String, dynamic>, t = config['thresholds'] as Map<String, dynamic>;
  num n(Map<String, dynamic> m, String k) => m[k] as num;
  final (g, ch) = channels(rgb, width, height);
  final (w, k) = reduce(g, n(a, 'work_width').toInt());
  final (cr, _) = reduce(ch, n(a, 'work_width').toInt());
  final bright = _mean(w);
  final glare = glareShare(w, n(a, 'glare_level').toDouble());
  final sharp = sharpness(w);
  final clip = n(a, 'grey_clip').toDouble();
  final grey = Plane(w.w, w.h);
  for (var i = 0; i < w.data.length; i++) {
    grey.data[i] = math.min(255.0 - w.data[i], clip);
  }
  double? period;
  var q = cr;
  var prof = <Float64List>[];
  for (final cand in [cr, grey]) {
    q = cand;
    prof = bandProfiles(cand,
        band: n(a, 'band').toInt(), detrend: n(a, 'detrend').toInt(), smooth: n(a, 'smooth').toInt());
    period = gridPeriod(prof,
        minLag: n(a, 'min_lag').toInt(),
        harmonics: n(a, 'harmonics').toInt(),
        combTol: n(a, 'comb_tol').toDouble());
    if (period != null) break;
  }
  final tilt = period == null ? null : tiltDeg(prof, period, band: n(a, 'band').toInt());
  final ppm = period == null ? null : period * k * scaleToCapture / 5.0;
  final (rows, blk) = paperBlocks(q, period, tilt, block: n(a, 'block').toInt(), ratio: n(a, 'paper_ratio').toDouble());
  var nBlocks = 0, nPaper = 0;
  for (final r in rows) {
    nBlocks += r.length;
    nPaper += r.where((v) => v).length;
  }
  final sheet = nBlocks == 0 ? 0.0 : nPaper / nBlocks;
  final group = largestGroup(closeBlocks(rows));
  final cw = sheetCorners(group, blk);
  final corners = cw?.map((p) => [_round1(p[0] * k * scaleToCapture), _round1(p[1] * k * scaleToCapture)]).toList();
  bool? whole;
  var area = 0.0;
  if (group.isNotEmpty) {
    final hb = rows.length, wb = rows[0].length, e = ((a['edge_blocks'] as num?) ?? 1).toInt();
    final on = [
      group.where((b) => b.$1 == 0).length,
      group.where((b) => b.$1 == hb - 1).length,
      group.where((b) => b.$2 == 0).length,
      group.where((b) => b.$2 == wb - 1).length,
    ];
    whole = on.reduce(math.max) < e;
    area = group.length / (hb * wb);
  }
  final shadow = shadowSpread(w, group, blk);
  final roi = sheetBrightness(w, group, blk);
  final hints = <String>[];
  if (ppm == null || ppm < n(t, 'min_px_per_mm')) hints.add('closer');
  if (glare > n(t, 'max_glare')) hints.add('glare');
  if (sharp < n(t, 'min_sharpness')) hints.add('steady');
  if (tilt != null && tilt.abs() > n(t, 'max_tilt_deg')) hints.add('level');
  if (bright < n(t, 'min_brightness')) hints.add('light');
  if ((roi ?? bright) > ((t['max_brightness'] as num?) ?? 255.0)) hints.add('bright');
  if (shadow != null && shadow > ((t['max_shadow'] as num?) ?? 1.0)) hints.add('shadow');
  if (area < n(t, 'min_sheet') || ((t['require_whole_sheet'] as bool?) ?? false) && whole == false) {
    hints.add('frame');
  }
  return FrameQc(
    width: width,
    height: height,
    factor: k,
    brightness: bright,
    glare: glare,
    sharpness: sharp,
    tiltDeg: tilt,
    periodPx: period,
    pxPerMm: ppm,
    sheet: sheet,
    hints: hints,
    shadow: shadow,
    corners: corners,
    sheetWhole: whole,
    sheetArea: area,
    sheetBrightness: roi,
  );
}
