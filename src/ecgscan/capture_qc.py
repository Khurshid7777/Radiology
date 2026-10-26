"""Frame quality checks for capture (TZ §6.1, SD-014, SD-046).

The same checks run on the phone (Dart, on preview frames) and on the server (a
re-check of the uploaded photo). To keep the two in step, every step is plain
arithmetic, specified here so that it can be re-implemented exactly, and both
implementations are tested against the same reference frames
(``tests/scanner/fixtures/capture_qc``). Thresholds live in
``configs/scanner/capture_qc.yaml`` and reach the client through
``GET /scans/config/capture-qc``.

Algorithm (all arithmetic in float64):

1. grey = 0.299 R + 0.587 G + 0.114 B and chroma = max(R, G, B) - min(R, G, B), both
   reduced by an integer factor ``k = ceil(width / work_width)`` with block means
   (the image cropped to a multiple of ``k``): ``w`` (grey) and ``c`` (chroma);
2. the grid channel ``q`` is ``c`` — the grid is coloured, the trace black, the paper
   white; on a preview the pale grid is invisible in grey next to traces and text.
   When steps 5-6 find no period on ``c`` (a grey grid; a coloured desk makes mean(c)
   useless as a test), they are repeated on ``q`` = min(255 - ``w``, ``grey_clip``): the
   clip keeps a black trace from outweighing the grey grid lines;
3. brightness = mean(w); glare = share of interior pixels whose 3×3 mean of ``w`` is
   >= ``glare_level`` (a glare spot is a blown-out plateau; scattered JPEG noise on
   bright paper is not);
4. sharpness = variance of the 4-neighbour Laplacian of ``w`` (interior pixels);
5. band profiles: ``q`` cut into consecutive bands of ``band`` rows (the rest
   dropped); a profile is the band's column means, minus its centred moving average
   of ``detrend`` px (edge-padded), then a centred moving average of ``smooth`` px
   (edge-padded; a fractional period must not lose to its whole-number multiples);
6. 5-mm period: A(l) = Σ over bands of Σ_x p(x) p(x + l) / n for l = 0 .. L,
   L = floor(0.4 n). A band's sideways shift (a tilted page) does not change its
   autocorrelation. At 5 mm both the 1-mm and 5-mm lines align. A comb picks the
   period, not one of its multiples: for l in [``min_lag``, L // 2],
   score(l) = mean over h = 1 .. H of max A[h l - h // 2 .. h l + h // 2],
   H = min(``harmonics``, L // l) (the window grows with h: a fractional period's
   multiples drift off whole lags). An exact period scores like its double, so m is
   the smallest lag with score >= ``comb_tol`` × the best score (half the period
   scores ~0.5: every other comb tooth falls between lines), then the argmax of A
   over m-1 .. m+1; no grid when A(m) <= 0.05 A(0).
   Refined by a parabola through A(m-1..m+1), offset clipped to ±0.5.
   px/mm = period × k × scale_to_capture / 5;
7. tilt: bands ``j`` = max(1, floor(1.5 × period / band)) apart are compared: the
   sideways shift s in [-S, S], S = floor(period / 2) - 1, maximising Σ p_i(x) p_{i+j}(x + s)
   (over the overlap, divided by n), refined by a parabola; tilt = atan(median(s) /
   (j × band)) in degrees (positive: lines lean right going down);
8. sheet: square blocks of B = max(``block``, 2 (P + |dP|)) working px of ``q``; a block is
   paper when the
   mean |q(x, y) - q(x + h, y + dh)| exceeds ``paper_ratio`` × the mean
   |q(x, y) - q(x + P, y + dP)| (P = round(period), h = P // 2; dP = -round(P tan(tilt)),
   dh = -round(h tan(tilt)) follow the tilted horizontal lines; as the server's paper
   mask at its period); sheet = share of paper blocks;
9. the sheet (SD-070, TZ addendum §18.2): the paper blocks closed on the block grid (a
   block is added when it, after setting every block that is paper or has a paper
   4-neighbour, has all four 4-neighbours set inside the grid: lone misread blocks inside
   the sheet are filled and the sheet never grows onto the frame edge), then the
   largest 4-connected group (row-major scan, the first of equal sizes). Its corners are the block corners with the
   least x + y (top left), the most x - y (top right), the most x + y (bottom right) and
   the least x - y (bottom left), the first in row-major block order and TL, TR, BR, BL
   point order on ties; in captured-photo px (× k × scale_to_capture). sheet_area =
   the group's blocks / all blocks. The sheet is whole when none of the four outer block
   lines (first / last row, first / last column) holds ``edge_blocks`` or more of its
   blocks (a misread desk block next to the sheet is one or two; a cut sheet runs along
   the edge);
10. shadow: over the sheet's inner blocks (all 4 neighbours in the sheet: an edge block is
   part desk), or all its blocks when no inner one is left. Their bounding box (in blocks)
   is cut into 4 × 4 zones, zone =
   floor(4 (b - b0) / n) per axis, a block's level is its mean ``w``; sheet_brightness =
   the mean of all the sheet's block levels (the TZ's "brightness of the ROI": a white
   sheet filling the frame is not over-exposed; the frame mean is used without a sheet);
   shadow = (max - min) / max over the zones with sheet blocks
   (``None`` below 2 zones). A zone's level is the 75th percentile of its block levels
   (sorted ascending, index floor(0.75 (n - 1))): a block half on the desk is darker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ecgscan.config import load_yaml

HINTS = ("closer", "glare", "steady", "level", "light", "bright", "shadow", "frame")


@dataclass
class FrameQC:
    width: int
    height: int
    factor: int
    brightness: float
    glare: float
    sharpness: float
    tilt_deg: float | None
    period_px: float | None          # 5-mm period on the working image
    px_per_mm: float | None          # on the captured photo
    sheet: float
    hints: list[str]
    ok: bool
    shadow: float | None = None      # brightness spread over 4 x 4 zones of the sheet
    corners: list[list[float]] | None = None   # sheet TL, TR, BR, BL in captured-photo px
    sheet_whole: bool | None = None  # the sheet does not run along an edge of the frame
    sheet_area: float = 0.0          # share of the frame the sheet covers (step 9)
    sheet_brightness: float | None = None   # mean grey of the sheet (step 10)

    def to_dict(self) -> dict:
        return asdict(self)


def config() -> dict:
    return load_yaml("capture_qc")


def channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(grey, chroma) of an H×W×3 RGB frame (the order a phone delivers)."""
    a = rgb.astype(np.float64)
    if a.ndim == 2:
        return a, np.zeros_like(a)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], a.max(axis=2) - a.min(axis=2)


def reduce(g: np.ndarray, work_width: int) -> tuple[np.ndarray, int]:
    k = max(1, math.ceil(g.shape[1] / work_width))
    h, w = (g.shape[0] // k) * k, (g.shape[1] // k) * k
    return g[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)), k


def sharpness(w: np.ndarray) -> float:
    lap = -4.0 * w[1:-1, 1:-1] + w[:-2, 1:-1] + w[2:, 1:-1] + w[1:-1, :-2] + w[1:-1, 2:]
    return float(lap.var())


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    c = np.cumsum(np.concatenate([[0.0], np.pad(x, (pad, win - 1 - pad), mode="edge")]))
    return (c[win:] - c[:-win]) / win


def band_profiles(q: np.ndarray, *, band: int, detrend: int, smooth: int) -> np.ndarray:
    rows = []
    for y in range(0, q.shape[0] - band + 1, band):
        p = q[y:y + band].mean(axis=0)
        rows.append(_moving_average(p - _moving_average(p, detrend), smooth))
    return np.array(rows).reshape(len(rows), q.shape[1])


def _parabola(y0: float, y1: float, y2: float) -> float:
    den = y0 - 2.0 * y1 + y2
    return min(0.5, max(-0.5, 0.5 * (y0 - y2) / den)) if den < 0 else 0.0


def grid_period(profiles: np.ndarray, *, min_lag: int, harmonics: int, comb_tol: float) -> float | None:
    if profiles.size == 0:
        return None
    n = profiles.shape[1]
    top = int(0.4 * n)
    if top // 2 <= min_lag + 1:
        return None
    a = np.array([float((profiles[:, : n - lag] * profiles[:, lag:]).sum()) / n for lag in range(top + 2)])
    if a[0] <= 0:
        return None
    lags = list(range(min_lag, top // 2 + 1))
    scores = []
    for lag in lags:
        h_max = min(harmonics, top // lag)
        scores.append(sum(float(a[h * lag - h // 2: h * lag + h // 2 + 1].max()) for h in range(1, h_max + 1)) / h_max)
    best = max(scores)
    m = next(lag for lag, sc in zip(lags, scores, strict=True) if sc >= comb_tol * best)
    m = m - 1 + int(np.argmax(a[m - 1: m + 2]))
    if a[m] <= 0.05 * a[0]:
        return None                                   # no periodic structure: no grid
    return float(m + _parabola(a[m - 1], a[m], a[m + 1]))


def tilt_deg(profiles: np.ndarray, period: float, *, band: int) -> float | None:
    j = max(1, int(1.5 * period / band))
    s_max = max(1, int(period / 2) - 1)
    n = profiles.shape[1]
    shifts = []
    for i in range(profiles.shape[0] - j):
        p, r = profiles[i], profiles[i + j]
        xc = np.array([float(np.dot(p[max(0, -s): n - max(0, s)], r[max(0, s): n - max(0, -s)])) / n
                       for s in range(-s_max - 1, s_max + 2)])
        k = 1 + int(np.argmax(xc[1:-1]))
        if xc[k] > 0:
            shifts.append(k - s_max - 1 + _parabola(xc[k - 1], xc[k], xc[k + 1]))
    if not shifts:
        return None
    return float(math.degrees(math.atan(float(np.median(shifts)) / (j * band))))


def _shift_diff(b: np.ndarray, dx: int, dy: int) -> float:
    """Mean |b(x, y) - b(x + dx, y + dy)| over the overlap (dx > 0)."""
    h = b.shape[0]
    y0, y1 = max(0, -dy), min(h, h - dy)
    return float(np.abs(b[y0:y1, : b.shape[1] - dx] - b[y0 + dy:y1 + dy, dx:]).mean())


def paper_blocks(q: np.ndarray, period: float | None, tilt: float | None, *, block: int,
                 ratio: float) -> tuple[list[list[bool]], int]:
    """Step 8: paper flag per block (rows of blocks) and the block size in working px."""
    if period is None:
        return [], block
    big = max(2, int(round(period)))
    half = max(1, big // 2)
    tan = math.tan(math.radians(tilt or 0.0))
    d_big, d_half = -int(round(big * tan)), -int(round(half * tan))
    block = max(block, 2 * (big + abs(d_big)))
    hb, wb = q.shape[0] // block, q.shape[1] // block
    rows = []
    for by in range(hb):
        row = []
        for bx in range(wb):
            b = q[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            row.append(_shift_diff(b, half, d_half) > ratio * _shift_diff(b, big, d_big) + 1e-9)
        rows.append(row)
    return (rows if wb > 0 else []), block


def sheet_fraction(q: np.ndarray, period: float | None, tilt: float | None, *, block: int, ratio: float) -> float:
    rows, _ = paper_blocks(q, period, tilt, block=block, ratio=ratio)
    n = sum(len(r) for r in rows)
    return 0.0 if n == 0 else sum(sum(r) for r in rows) / n


def close_blocks(rows: list[list[bool]]) -> list[list[bool]]:
    """Step 9: paper blocks OR their closing (dilation, then erosion with outside the grid unset)."""
    hb = len(rows)
    wb = len(rows[0]) if hb else 0
    nb = ((-1, 0), (1, 0), (0, -1), (0, 1))
    dil = [[rows[y][x] or any(0 <= y + dy < hb and 0 <= x + dx < wb and rows[y + dy][x + dx] for dy, dx in nb)
            for x in range(wb)] for y in range(hb)]
    return [[rows[y][x] or (dil[y][x] and all(0 <= y + dy < hb and 0 <= x + dx < wb and dil[y + dy][x + dx]
                                              for dy, dx in nb))
             for x in range(wb)] for y in range(hb)]


def largest_group(rows: list[list[bool]]) -> list[tuple[int, int]]:
    """Step 9: the largest 4-connected group of True blocks, as (by, bx) in visit order."""
    hb = len(rows)
    wb = len(rows[0]) if hb else 0
    seen = [[False] * wb for _ in range(hb)]
    best: list[tuple[int, int]] = []
    for y in range(hb):
        for x in range(wb):
            if not rows[y][x] or seen[y][x]:
                continue
            group, stack = [], [(y, x)]
            seen[y][x] = True
            while stack:
                cy, cx = stack.pop()
                group.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < hb and 0 <= nx < wb and rows[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(group) > len(best):
                best = group
    return sorted(best)


def sheet_corners(group: list[tuple[int, int]], block: int) -> list[tuple[float, float]] | None:
    """Step 9: TL, TR, BR, BL of the group's block corners (working px)."""
    if not group:
        return None
    keys = (lambda x, y: -(x + y), lambda x, y: x - y, lambda x, y: x + y, lambda x, y: y - x)
    best: list[tuple[float, float] | None] = [None] * 4
    score = [-math.inf] * 4
    for by, bx in group:
        for x, y in ((bx * block, by * block), ((bx + 1) * block, by * block),
                     ((bx + 1) * block, (by + 1) * block), (bx * block, (by + 1) * block)):
            for i, f in enumerate(keys):
                v = f(x, y)
                if v > score[i]:
                    score[i], best[i] = v, (float(x), float(y))
    return best  # type: ignore[return-value]


def sheet_brightness(w: np.ndarray, group: list[tuple[int, int]], block: int) -> float | None:
    """Step 10: the mean of the sheet's block levels."""
    if not group:
        return None
    return sum(float(w[by * block:(by + 1) * block, bx * block:(bx + 1) * block].mean()) for by, bx in group) / len(group)


def shadow_spread(w: np.ndarray, group: list[tuple[int, int]], block: int) -> float | None:
    """Step 10: (max - min) / max of the zone levels over a 4 x 4 split of the sheet."""
    if not group:
        return None
    members = set(group)
    inner = [(by, bx) for by, bx in group
             if {(by - 1, bx), (by + 1, bx), (by, bx - 1), (by, bx + 1)} <= members]
    group = inner or group
    y0, y1 = min(b[0] for b in group), max(b[0] for b in group)
    x0, x1 = min(b[1] for b in group), max(b[1] for b in group)
    ny, nx = y1 - y0 + 1, x1 - x0 + 1
    zones: dict[tuple[int, int], list[float]] = {}
    for by, bx in group:
        level = float(w[by * block:(by + 1) * block, bx * block:(bx + 1) * block].mean())
        zones.setdefault((4 * (by - y0) // ny, 4 * (bx - x0) // nx), []).append(level)
    if len(zones) < 2:
        return None
    levels = [sorted(v)[int(0.75 * (len(v) - 1))] for v in zones.values()]
    top = max(levels)
    return 0.0 if top <= 0 else (top - min(levels)) / top


def check_frame(rgb: np.ndarray, *, scale_to_capture: float = 1.0, cfg: dict | None = None,
                grid: bool = True) -> FrameQC:
    """QC of one frame. ``scale_to_capture`` = captured photo width / this frame's width.

    ``grid=False`` skips steps 5-8 (period, tilt, sheet: ~2 s on a full photo in Python):
    the server records only brightness, glare and sharpness."""
    c = cfg or config()
    a, t = c["algorithm"], c["thresholds"]
    if grid:
        g, ch = channels(rgb)
        w, k = reduce(g, a["work_width"])
        cr, _ = reduce(ch, a["work_width"])
    else:
        # Without the grid steps chroma is not needed and grey is linear in RGB: reduce
        # the bytes first (OpenCV's box mean, exact for an integer factor), then grey.
        import cv2

        k = max(1, math.ceil(rgb.shape[1] / a["work_width"]))
        hh, ww = rgb.shape[0] // k, rgb.shape[1] // k
        small = cv2.resize(np.ascontiguousarray(rgb[: hh * k, : ww * k]).astype(np.float32), (ww, hh),
                           interpolation=cv2.INTER_AREA)
        w, _ = channels(small)
        g, cr = w, w
    bright = float(w.mean())
    m3 = sum(w[1 + dy: w.shape[0] - 1 + dy, 1 + dx: w.shape[1] - 1 + dx]
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    glare = float((m3 >= a["glare_level"]).mean())
    sharp = sharpness(w)
    period, prof, q = None, np.zeros((0, 0)), cr
    for q in (cr, np.minimum(255.0 - w, a["grey_clip"])) if grid else ():
        prof = band_profiles(q, band=a["band"], detrend=a["detrend"], smooth=a["smooth"])
        period = grid_period(prof, min_lag=a["min_lag"], harmonics=a["harmonics"], comb_tol=a["comb_tol"])
        if period is not None:
            break
    tilt = None if period is None else tilt_deg(prof, period, band=a["band"])
    ppm = None if period is None else period * k * scale_to_capture / 5.0
    rows, blk = paper_blocks(q, period, tilt, block=a["block"], ratio=a["paper_ratio"])
    n_blocks = sum(len(r) for r in rows)
    sheet = 0.0 if n_blocks == 0 else sum(sum(r) for r in rows) / n_blocks
    group = largest_group(close_blocks(rows))
    corners_w = sheet_corners(group, blk)
    corners = None if corners_w is None else [[round(x * k * scale_to_capture, 1), round(y * k * scale_to_capture, 1)]
                                              for x, y in corners_w]
    whole, area = None, 0.0
    if group:
        hb, wb, e = len(rows), len(rows[0]), int(a.get("edge_blocks", 1))
        on = [sum(1 for by, _ in group if by == 0), sum(1 for by, _ in group if by == hb - 1),
              sum(1 for _, bx in group if bx == 0), sum(1 for _, bx in group if bx == wb - 1)]
        whole = max(on) < e
        area = len(group) / (hb * wb)
    shadow = shadow_spread(w, group, blk)
    roi = sheet_brightness(w, group, blk)
    hints = []
    if ppm is None or ppm < t["min_px_per_mm"]:
        hints.append("closer")
    if glare > t["max_glare"]:
        hints.append("glare")
    if sharp < t["min_sharpness"]:
        hints.append("steady")
    if tilt is not None and abs(tilt) > t["max_tilt_deg"]:
        hints.append("level")
    if bright < t["min_brightness"]:
        hints.append("light")
    if (bright if roi is None else roi) > t.get("max_brightness", 255.0):
        hints.append("bright")
    if shadow is not None and shadow > t.get("max_shadow", 1.0):
        hints.append("shadow")
    if area < t["min_sheet"] or (t.get("require_whole_sheet", False) and whole is False):
        hints.append("frame")
    return FrameQC(width=int(rgb.shape[1]), height=int(rgb.shape[0]), factor=k, brightness=round(bright, 3),
                   glare=round(glare, 5), sharpness=round(sharp, 3),
                   tilt_deg=None if tilt is None else round(tilt, 3),
                   period_px=None if period is None else round(period, 3),
                   px_per_mm=None if ppm is None else round(ppm, 3), sheet=round(sheet, 4), hints=hints,
                   ok=not hints, shadow=None if shadow is None else round(shadow, 4), corners=corners,
                   sheet_whole=whole, sheet_area=round(area, 4),
                   sheet_brightness=None if roi is None else round(roi, 3))
