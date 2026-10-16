"""Grid pitch (px per mm) from the paper's 1-mm / 5-mm lattice (TZ §5.3).

The legacy estimator (``cardiolens...digitize.estimate_grid_pitch``) picks the
strongest autocorrelation peak and on photos often lands on a harmonic: 6 instead
of 12 px/mm, or 4 / 8 (M1, SD-022). Here the ECG grid's own structure decides:
the image is periodic with the *major* period M = 5 * pitch, so its spectrum has
teeth at every k / M. For a candidate pitch the comb score is the mean spectral
magnitude on the teeth minus the mean half-way between them:

* the true pitch: teeth on harmonics, half-way points empty -> high;
* half the pitch (6 vs 12): half-way points land on the true odd harmonics -> penalised;
* double / 2/3 / 3/2 pitch: many teeth fall between harmonics -> lower mean.

The grid is taken from the chroma channel (max - min of BGR: a coloured grid on
white paper, the black trace and the paper itself have ~0 chroma) and, for a grey
grid, from the inverted brightness; the channel with the better comb wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridEstimate:
    px_per_mm_x: float | None
    px_per_mm_y: float | None
    score_x: float
    score_y: float
    channel: str
    confidence: float

    @property
    def px_per_mm(self) -> float | None:
        if self.px_per_mm_x is None or self.px_per_mm_y is None:
            return None
        return float(np.sqrt(self.px_per_mm_x * self.px_per_mm_y))


def _channels(img: np.ndarray) -> dict[str, np.ndarray]:
    if img.ndim == 2:
        return {"gray": 255.0 - img.astype(np.float32)}
    f = img.astype(np.float32)
    return {"chroma": f.max(axis=2) - f.min(axis=2), "gray": 255.0 - f.mean(axis=2)}


def _spectrum(ch: np.ndarray, axis: int, max_rows: int = 400) -> np.ndarray:
    """Mean |FFT| of the detrended profiles along ``axis`` (phase-free, perspective-tolerant)."""
    a = ch if axis == 1 else ch.T
    step = max(1, a.shape[0] // max_rows)
    rows = a[::step].astype(np.float64)
    rows = rows - rows.mean(axis=1, keepdims=True)
    win = np.hanning(rows.shape[1])
    return np.abs(np.fft.rfft(rows * win, axis=1)).mean(axis=0)


def _sample(spec: np.ndarray, n: int, freq: float, tol_bins: float) -> float:
    """Max magnitude within ±tol bins of ``freq`` (cycles/px)."""
    b = freq * n
    lo, hi = int(np.floor(b - tol_bins)), int(np.ceil(b + tol_bins)) + 1
    lo, hi = max(1, lo), min(len(spec), hi)
    return float(spec[lo:hi].max()) if hi > lo else 0.0


def comb_score(spec: np.ndarray, n: int, pitch_px: float, *, max_k: int = 12, rel_tol: float = 0.01) -> float:
    major = 5.0 * pitch_px
    teeth, gaps = [], []
    for k in range(1, max_k + 1):
        f = k / major
        if f >= 0.5:
            break
        tol = max(1.0, rel_tol * f * n)               # perspective spreads high harmonics
        teeth.append(_sample(spec, n, f, tol))
        gaps.append(_sample(spec, n, (k - 0.5) / major, tol))
    if len(teeth) < 3:
        return 0.0
    base = float(np.median(spec[1:])) + 1e-9
    return (float(np.mean(teeth)) - float(np.mean(gaps))) / base


_WIDTHS = (3, 5, 9, 17, 33)


def _maxed(spec: np.ndarray) -> dict[int, np.ndarray]:
    from scipy.ndimage import maximum_filter1d

    return {w: maximum_filter1d(spec, size=w, mode="nearest") for w in _WIDTHS}


def comb_scores(spec: np.ndarray, n: int, pitches: np.ndarray, *, max_k: int = 12, rel_tol: float = 0.01,
                maxed: dict[int, np.ndarray] | None = None, explained: bool = True) -> np.ndarray:
    """Vectorised ``comb_score`` for many candidate pitches (same definition)."""
    maxed = maxed or _maxed(spec)
    p = np.asarray(pitches, dtype=np.float64)[:, None]
    k = np.arange(1, max_k + 1, dtype=np.float64)[None, :]
    f_t, f_g = k / (5.0 * p), (k - 0.5) / (5.0 * p)
    valid = f_t < 0.5
    tol = np.maximum(1.0, rel_tol * f_t * n)
    widths = np.array(_WIDTHS)
    wi = np.abs(widths[None, None, :] - (2 * tol + 1)[:, :, None]).argmin(axis=2)
    stack = np.stack([maxed[w] for w in _WIDTHS])            # [n_widths, n_bins]
    last = len(spec) - 1

    def sample(f):
        b = np.clip(np.rint(f * n).astype(int), 1, last)
        return stack[wi, b]

    t, g = sample(f_t), sample(f_g)
    cnt = valid.sum(axis=1)
    mean_t = np.where(cnt > 0, (t * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    mean_g = np.where(cnt > 0, (g * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    base = float(np.median(spec[1:])) + 1e-9
    score = np.where(cnt >= 3, (mean_t - mean_g) / base, 0.0)
    # The 1-mm lines themselves must show: energy at 1/pitch. Without it a sharp
    # lattice's harmonics also fit a comb at 2/5 of the pitch (8 px/mm read as 3.2).
    minor = maxed[9][np.clip(np.rint(n / p[:, 0]).astype(int), 1, last)] / base
    score = np.where(minor >= MINOR_MIN, score, score * 0.25)
    return score * explained_fraction(spec, n, p[:, 0]) ** 2 if explained else score


def _peaks(spec: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import find_peaks

    base = float(np.median(spec[1:])) + 1e-9
    idx, _ = find_peaks(spec, prominence=PEAK_PROMINENCE * base)
    idx = idx[idx > 2]
    return idx / n, spec[idx]


def explained_fraction(spec: np.ndarray, n: int, pitches: np.ndarray) -> np.ndarray:
    """Share of the strong spectral peaks' magnitude lying on multiples of 1/(5 * pitch).

    The true grid is the *largest* major period explaining every strong peak: a comb at
    pitch/2 or pitch/3 sits on harmonics too (sharp lines have them all) but leaves the
    major fundamental and its low multiples unexplained.
    """
    f, a = _peaks(spec, n)
    if f.size == 0:
        return np.ones(len(pitches))
    x = f[None, :] * 5.0 * np.asarray(pitches)[:, None]          # harmonic index per peak
    tol = np.maximum(0.08, 1.5 / (5.0 * np.asarray(pitches)[:, None]))   # ±1.5 px of the major period, at least 8 %
    on = np.abs(x - np.rint(x)) <= tol
    on &= np.rint(x) >= 1
    return (on * a[None, :]).sum(axis=1) / a.sum()


def _best_pitch(spec: np.ndarray, n: int, lo: float, hi: float) -> tuple[float | None, float]:
    """Coarse log-spaced search, divisor check, then a fine search around the winner."""
    maxed = _maxed(spec)
    cands = np.exp(np.linspace(np.log(lo), np.log(hi), 400))
    scores = comb_scores(spec, n, cands, maxed=maxed)
    i = int(np.argmax(scores))
    if scores[i] <= 0:
        return None, 0.0
    best, best_s = float(cands[i]), float(scores[i])
    # A comb at 2x / 3x the pitch keeps the strong low harmonics and can win when the
    # high ones fade (blur, perspective); the true pitch still explains it, so prefer
    # a divisor that scores at least half as well (a wrong divisor scores < 0 on its gaps).
    for d in (2, 3):
        if best / d >= lo:
            sd = float(comb_scores(spec, n, best / d * np.linspace(0.98, 1.02, 9), maxed=maxed).max())
            if sd >= SUBHARMONIC_ACCEPT * best_s:
                best, best_s = best / d, sd
                break
    return _refine(spec, n, best, maxed=maxed), best_s


def _refine(spec: np.ndarray, n: int, p: float, span: float = 0.02, maxed=None) -> float:
    fine = np.linspace(p * (1 - span), p * (1 + span), 81)
    fs = comb_scores(spec, n, fine, max_k=25, rel_tol=0.004, maxed=maxed)
    return float(fine[int(np.argmax(fs))])


PEAK_PROMINENCE = 8.0         # spectral peaks this many medians above the floor must be explained
MINOR_MIN = 3.0               # spectral magnitude at the 1-mm frequency, in units of the median
SUBHARMONIC_ACCEPT = 0.5      # a divisor pitch is preferred when it scores at least this fraction
CONF_SCALE = 3.0              # comb score (in spectral medians) that counts as full confidence
SQUARE_TOL = 0.15             # x and y pitch may differ this much (perspective, non-square pixels)


def estimate_grid(img: np.ndarray, *, min_px_per_mm: float = 2.5, max_px_per_mm: float = 40.0,
                  min_score: float = 0.2) -> GridEstimate:
    """Grid pitch along x and y; ``None`` where no lattice is found."""
    best: GridEstimate | None = None
    for name, ch in _channels(img).items():
        res = []
        for axis in (1, 0):
            spec = _spectrum(ch, axis)
            n = ch.shape[axis]
            hi = min(max_px_per_mm, n / 60.0)          # at least a dozen major periods in view
            p, s = _best_pitch(spec, n, min_px_per_mm, hi) if hi > min_px_per_mm else (None, 0.0)
            res.append((p if s >= min_score else None, s))
        (px, sx), (py, sy) = res
        if px is not None and py is not None and abs(px - py) > SQUARE_TOL * max(px, py):
            # The paper grid is square: re-read the weaker axis near the stronger one's pitch.
            specs = {1: _spectrum(ch, 1), 0: _spectrum(ch, 0)}
            strong_p, weak_axis = (px, 0) if sx >= sy else (py, 1)
            n_w = ch.shape[weak_axis]
            grid_f = strong_p * np.linspace(0.9, 1.1, 41)
            # Raw comb here: along x the heartbeat's own periodicity adds peaks no grid explains.
            sc = comb_scores(specs[weak_axis], n_w, grid_f, explained=False)
            sw, pw = float(sc.max()), float(grid_f[int(sc.argmax())])
            weak_p = py if weak_axis == 0 else px
            own = float(comb_scores(specs[weak_axis], n_w, np.array([weak_p]), explained=False)[0])
            if sw >= SUBHARMONIC_ACCEPT * own:
                pw = _refine(specs[weak_axis], n_w, pw)
                if weak_axis == 0:
                    py, sy = pw, sw
                else:
                    px, sx = pw, sw
        conf = float(np.clip(min(sx, sy) / CONF_SCALE, 0.0, 1.0))
        if px is not None and py is not None and abs(px - py) > 0.15 * max(px, py):
            conf *= 0.3                                 # x and y must describe the same square grid
        est = GridEstimate(px, py, sx, sy, name, round(conf, 3))
        # Chroma first: a coloured grid read there is not confused by the black trace rows
        # (a 12x1 strip's row pitch is a strong lattice in brightness). Brightness is the
        # fallback for grey grids only.
        if name == "chroma" and est.px_per_mm is not None:
            return est
        if best is None or min(sx, sy) > min(best.score_x, best.score_y):
            best = est
    return best  # type: ignore[return-value]
