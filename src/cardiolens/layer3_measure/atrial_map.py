"""Conducted P waves of one shape at a constant PR (D358; the first piece of the atrial map, D357).

P in place comes from the D295 detector (``p_cnn``) in II, V1, III and aVF, merged across leads (within 60 ms).
For each QRS after the first, the latest P 80-400 ms before it is its conducted P. Reported: the share of QRS with
such a P, the PR spread (IQR) and how alike the conducted P are — the median correlation of each P (+-60 ms, 1-25 Hz)
with their median, in II and in V1. Atrial fibrillation gives "P" at a constant distance too (the QRS onset), but of
varying shape; a sinus or ectopic rhythm with extrasystoles gives the same P before most QRS.
"""
from __future__ import annotations

import numpy as np

LEADS = ("II", "V1", "III", "aVF")
MERGE_S = 0.06
PR_RANGE_S = (0.08, 0.40)
SHAPE_HALF_S = 0.06
SHAPE_BAND_HZ = (1.0, 25.0)


def _in_place_p(leads: dict[str, np.ndarray], fs: float) -> np.ndarray:
    from cardiolens.layer3_measure.p_cnn import p_peaks

    ps: list[int] = []
    for ln in LEADS:
        x = leads.get(ln)
        if x is None:
            continue
        p = p_peaks(np.asarray(x, float), fs)
        if p is not None:
            ps.extend(int(v) for v in p)
    if not ps:
        return np.array([], int)
    ps.sort()
    groups: list[list[int]] = []
    for t in ps:
        if groups and t - groups[-1][-1] <= MERGE_S * fs:
            groups[-1].append(t)
        else:
            groups.append([t])
    return np.asarray([float(np.median(g)) for g in groups], float)


def conducted_p(leads: dict[str, np.ndarray], rpeaks, fs: float) -> dict | None:
    """``share`` of QRS with a conducted P, ``pr_iqr_s``, ``shape_ii`` / ``shape_v1`` (median correlation; None when
    the lead is missing or fewer than 3 P); None with fewer than 4 QRS or no P detector."""
    from scipy.signal import butter, filtfilt

    rp = np.asarray(rpeaks if rpeaks is not None else [], int)
    if rp.size < 5:
        return None
    p = _in_place_p(leads, fs)
    pos, prs = [], []
    for q in rp[1:]:
        c = p[(p < q - PR_RANGE_S[0] * fs) & (p > q - PR_RANGE_S[1] * fs)]
        if c.size:
            pos.append(int(c.max()))
            prs.append((q - c.max()) / fs)
    out = {"share": len(prs) / (rp.size - 1), "n_p": int(p.size),
           "pr_iqr_s": float(np.subtract(*np.percentile(prs, [75, 25]))) if len(prs) >= 3 else None,
           "shape_ii": None, "shape_v1": None}
    w = int(SHAPE_HALF_S * fs)
    nyq = fs / 2.0
    b, a = butter(2, [SHAPE_BAND_HZ[0] / nyq, min(SHAPE_BAND_HZ[1] / nyq, 0.99)], btype="band")
    for ln, key in (("II", "shape_ii"), ("V1", "shape_v1")):
        x = leads.get(ln)
        if x is None or len(pos) < 3 or len(x) < 3 * w:
            continue
        y = filtfilt(b, a, np.asarray(x, float))
        segs = np.array([y[t - w:t + w] for t in pos if t - w >= 0 and t + w <= len(y)])
        if len(segs) < 3:
            continue
        tpl = np.median(segs, axis=0)
        cors = [np.corrcoef(s, tpl)[0, 1] for s in segs if np.std(s) > 0]
        if cors and np.std(tpl) > 0:
            out[key] = float(np.median(cors))
    return out
