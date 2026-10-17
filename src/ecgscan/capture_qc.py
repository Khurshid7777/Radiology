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
   mask at its period); sheet = share of paper blocks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ecgscan.config import load_yaml

HINTS = ("closer", "glare", "steady", "level", "light", "frame")


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


def sheet_fraction(q: np.ndarray, period: float | None, tilt: float | None, *, block: int, ratio: float) -> float:
    if period is None:
        return 0.0
    big = max(2, int(round(period)))
    half = max(1, big // 2)
    tan = math.tan(math.radians(tilt or 0.0))
    d_big, d_half = -int(round(big * tan)), -int(round(half * tan))
    block = max(block, 2 * (big + abs(d_big)))
    hb, wb = q.shape[0] // block, q.shape[1] // block
    if hb == 0 or wb == 0:
        return 0.0
    paper = 0
    for by in range(hb):
        for bx in range(wb):
            b = q[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            if _shift_diff(b, half, d_half) > ratio * _shift_diff(b, big, d_big) + 1e-9:
                paper += 1
    return paper / (hb * wb)


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
    sheet = sheet_fraction(q, period, tilt, block=a["block"], ratio=a["paper_ratio"])
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
    if sheet < t["min_sheet"]:
        hints.append("frame")
    return FrameQC(width=int(rgb.shape[1]), height=int(rgb.shape[0]), factor=k, brightness=round(bright, 3),
                   glare=round(glare, 5), sharpness=round(sharp, 3),
                   tilt_deg=None if tilt is None else round(tilt, 3),
                   period_px=None if period is None else round(period, 3),
                   px_per_mm=None if ppm is None else round(ppm, 3), sheet=round(sheet, 4), hints=hints,
                   ok=not hints)
