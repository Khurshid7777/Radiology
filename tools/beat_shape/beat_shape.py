"""Per-beat "ventricular or not" by QRS shape and timing (D314, plan 6.4).

Features per beat from two leads (II and V1 when present, else the first two of the run detector's leads): correlation
with the narrow-QRS template (D307) and with the all-beats template, QRS width and its excess over the template's,
amplitude and area relative to the template, a polarity flip; plus the preceding / following RR over the reference
cycle. The model (models/beat_shape/beat_v_hgb.joblib, gradient boosting trained on even INCART records only) gives
P(ventricular). Without the model file every call returns None and the run detector works as before (D307).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[3] / "models" / "beat_shape" / "beat_v_hgb.joblib"
HALF_S = 0.10
NARROW_MS = 20.0
LEADS = ("II", "V1")
N_FEAT = 2 * 7 + 2


def _widths(segs: np.ndarray, fs: float) -> np.ndarray:
    d = np.abs(np.diff(segs, axis=1))
    env = d / (d.max(axis=1, keepdims=True) + 1e-9)
    out = np.empty(len(segs))
    for k, e in enumerate(env):
        above = np.flatnonzero(e >= 0.3)
        out[k] = (above[-1] - above[0]) / fs * 1000.0 if above.size >= 2 else 0.0
    return out


def _corr(segs: np.ndarray, t: np.ndarray) -> np.ndarray:
    a = segs - segs.mean(axis=1, keepdims=True)
    b = t - t.mean()
    return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)


def features(leads: dict[str, np.ndarray], rpeaks, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(feature matrix, indices into ``rpeaks`` of the beats it covers)."""
    rp = np.asarray(rpeaks, int)
    names = [n for n in LEADS if n in leads] or list(leads)[:2]
    half = int(HALF_S * fs)
    n_s = min(len(leads[n]) for n in names) if names else 0
    ok = np.array([i for i, r in enumerate(rp) if r - half >= 0 and r + half < n_s], int)
    if len(ok) < 4 or not names:
        return np.empty((0, N_FEAT)), ok
    segs = {n: np.stack([np.asarray(leads[n], float)[rp[i] - half:rp[i] + half] for i in ok]) for n in names}
    # the narrow template (D307): beats within NARROW_MS of the narrowest QRS (summed over the leads)
    wsum = np.mean([_widths(segs[n], fs) for n in names], axis=0)
    narrow = np.flatnonzero(wsum <= wsum.min() + NARROW_MS)
    if narrow.size < 2:
        narrow = np.arange(len(ok))
    cols = []
    for k in range(2):
        n = names[min(k, len(names) - 1)]
        s = segs[n]
        tn, ta = np.median(s[narrow], axis=0), np.median(s, axis=0)
        w = _widths(s, fs)
        wt = float(_widths(tn[None, :], fs)[0])
        ptp_t = float(np.ptp(tn)) + 1e-9
        area_t = float(np.abs(tn - np.median(tn)).sum()) + 1e-9
        pol = np.sign(s[np.arange(len(s)), np.argmax(np.abs(s - np.median(s, axis=1, keepdims=True)), axis=1)]
                      - np.median(s, axis=1))
        pol_t = np.sign(tn[np.argmax(np.abs(tn - np.median(tn)))] - np.median(tn))
        cols += [_corr(s, tn), _corr(s, ta), w, w - wt, np.ptp(s, axis=1) / ptp_t,
                 np.abs(s - np.median(s, axis=1, keepdims=True)).sum(axis=1) / area_t, (pol == pol_t).astype(float)]
    rr = np.diff(rp).astype(float)
    ends = [rr[ok[k] - 1] for k in narrow if ok[k] >= 1]
    ref = float(np.median(ends)) if ends else float(np.percentile(rr, 75))
    ref = ref or 1.0
    pre = np.array([rr[i - 1] / ref if i >= 1 else np.nan for i in ok])
    post = np.array([rr[i] / ref if i < len(rr) else np.nan for i in ok])
    cols += [pre, post]
    return np.column_stack(cols), ok


@lru_cache(maxsize=1)
def _model():
    if not MODEL.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL)
    except Exception:
        return None


def p_ventricular(leads: dict[str, np.ndarray], rpeaks, fs: float) -> np.ndarray | None:
    """P(ventricular) for every beat in ``rpeaks`` (NaN where no window); None without the model."""
    model = _model()
    if model is None:
        return None
    X, ok = features(leads, rpeaks, fs)
    out = np.full(len(rpeaks), np.nan)
    if len(ok):
        out[ok] = model.predict_proba(X)[:, 1]
    return out
