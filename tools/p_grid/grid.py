"""D313: a regular atrial grid over P-model probabilities (100 Hz traces from dump.py)."""
from __future__ import annotations

import numpy as np

FS = 100


def best_grid(prob: np.ndarray, r100, qrs_ms: float | None, tmin=0.50, tmax=1.20, win=4, avoid=()):
    """Best (period s, phase, z, visible count, grid positions, mean). For every period the phases are compared with
    each other (z of the best phase among all phases of that period), so a long period with few points is not
    favoured by chance."""
    from scipy.ndimage import maximum_filter1d
    p = np.asarray(prob, float)
    n = len(p)
    mx = maximum_filter1d(p, size=2 * win + 1)
    blank = np.zeros(n, bool)
    half = max(0.06, (qrs_ms or 100) / 2000 + 0.03)
    for r in r100:
        blank[max(0, int(r - half * FS)):int(r + (half + 0.04) * FS)] = True
    best = None
    for per in range(int(tmin * FS), int(tmax * FS) + 1):
        if any(abs(per / FS - a) <= 0.08 * a for a in avoid):     # locked to the QRS (leftover QRST): not atrial
            continue
        scores = []
        for ph in range(per):
            pos = np.arange(ph, n, per)
            vis = pos[~blank[pos]]
            scores.append(float(mx[vis].mean()) if vis.size >= 6 else np.nan)
        sc = np.asarray(scores)
        if np.isnan(sc).all():
            continue
        ph = int(np.nanargmax(sc))
        z = (sc[ph] - np.nanmean(sc)) / (np.nanstd(sc) + 1e-9)
        if best is None or z > best[2]:
            pos = np.arange(ph, n, per)
            best = (per / FS, ph, float(z), int((~blank[pos]).sum()), pos, float(sc[ph]))
    return best
def features(prob, r100, qrs_ms, hr):
    rr = np.diff(np.asarray(r100, float)) / FS
    avoid = () if rr.size == 0 else (float(np.median(rr)), float(np.median(rr)) / 2, float(np.median(rr)) / 3)
    g = best_grid(prob, r100, qrs_ms, avoid=avoid)
    if g is None or hr is None:
        return None
    per, ph, sc, nvis, pos, mean = g
    r = np.asarray(r100, int)
    prs = []
    for q in r:
        before = pos[(pos < q - 4) & (pos > q - int(1.2 * per * FS))]
        if before.size:
            prs.append((q - before[-1]) / FS)
    iqr = float(np.subtract(*np.percentile(prs, [75, 25]))) if len(prs) >= 4 else None
    return {"rate": 60.0 / per, "score": sc, "mean": mean, "nvis": nvis, "ratio": 60.0 / per / hr, "pr_iqr": iqr}
