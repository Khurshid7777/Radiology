"""Complete AV block by a regular atrial grid on the QRST-subtracted residual (D313 / D316).

After the median QRST complex is subtracted, P waves locked to the QRS vanish and dissociated P waves stay. The
residual's P-band envelope (1.5-12 Hz, in robust SDs, at 100 Hz) is searched for the regular grid (period 0.50-1.20 s,
any phase) whose points stand out most against the other phases of the same period; periods at RR, RR/2 and RR/3 are
skipped (leftover QRST). Thresholds were chosen on the even half of the Chapman bench (D313) and frozen (D316).
"""
from __future__ import annotations

import numpy as np

FS = 100
PERIOD_S = (0.50, 1.20)
LOCKED_TOL = 0.08
MIN_VISIBLE = 6
Z_MIN = 3.0
MEAN_MIN = 6.0
RATIO_MIN = 1.3
PR_IQR_MIN_S = 0.10
LEADS = ("II", "V1")


def envelope(residual: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, resample_poly
    b, a = butter(2, [1.5 / (fs / 2), 12.0 / (fs / 2)], btype="band")
    y = np.abs(filtfilt(b, a, np.asarray(residual, float)))
    y = resample_poly(y, FS, int(round(fs)))
    return y / (1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9)


def grid_features(env: np.ndarray, r100: np.ndarray, qrs_ms: float | None, hr: float | None) -> dict | None:
    from scipy.ndimage import maximum_filter1d
    r100 = np.asarray(r100, int)
    if hr is None or r100.size < 3:
        return None
    n = len(env)
    mx = maximum_filter1d(np.asarray(env, float), size=9)
    blank = np.zeros(n, bool)
    half = max(0.06, (qrs_ms or 100) / 2000 + 0.03)
    for r in r100:
        blank[max(0, int(r - half * FS)):int(r + (half + 0.04) * FS)] = True
    rr = float(np.median(np.diff(r100))) / FS
    avoid = (rr, rr / 2, rr / 3)
    best = None
    for per in range(int(PERIOD_S[0] * FS), int(PERIOD_S[1] * FS) + 1):
        if any(abs(per / FS - a) <= LOCKED_TOL * a for a in avoid):
            continue
        sc = np.full(per, np.nan)
        for ph in range(per):
            pos = np.arange(ph, n, per)
            vis = pos[~blank[pos]]
            if vis.size >= MIN_VISIBLE:
                sc[ph] = mx[vis].mean()
        if np.isnan(sc).all():
            continue
        ph = int(np.nanargmax(sc))
        z = (sc[ph] - np.nanmean(sc)) / (np.nanstd(sc) + 1e-9)
        if best is None or z > best[0]:
            best = (float(z), per, ph, float(sc[ph]))
    if best is None:
        return None
    z, per, ph, mean = best
    pos = np.arange(ph, n, per)
    prs = []
    for q in r100:
        before = pos[(pos < q - 4) & (pos > q - int(1.2 * per))]
        if before.size:
            prs.append((q - before[-1]) / FS)
    iqr = float(np.subtract(*np.percentile(prs, [75, 25]))) if len(prs) >= 4 else None
    return {"rate": 60.0 * FS / per, "z": z, "mean": mean, "ratio": 60.0 * FS / per / hr, "pr_iqr_s": iqr}


def dissociated(f: dict | None) -> bool:
    return bool(f and f["pr_iqr_s"] is not None and f["z"] >= Z_MIN and f["mean"] >= MEAN_MIN
                and f["ratio"] >= RATIO_MIN and f["pr_iqr_s"] >= PR_IQR_MIN_S)


def grid_block(leads: dict[str, np.ndarray], rpeaks, fs: float, qrs_ms: float | None, hr: float | None) -> dict | None:
    """The first lead (II, then V1) whose atrial grid reads as AV dissociation; None otherwise."""
    from cardiolens.layer3_measure.atrial import template_subtract
    rp = np.asarray(rpeaks, int)
    if rp.size < 4:
        return None
    r100 = np.round(rp * FS / fs).astype(int)
    for ln in LEADS:
        x = leads.get(ln)
        if x is None:
            continue
        res = template_subtract(np.asarray(x, float), rp, fs, 0.30, 0.55)
        if res is None:
            continue
        f = grid_features(envelope(res, fs), r100, qrs_ms, hr)
        if dissociated(f):
            return dict(f, lead=ln)
    return None
