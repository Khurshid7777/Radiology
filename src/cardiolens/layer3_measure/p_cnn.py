"""P waves by the trained detector (D295; accepted for use outside AF / atrial flutter / pacing only).

A small 1-D CNN (models/p_detector/p_cnn_d295_it2.onnx, 150 KB, onnxruntime) marks "inside a P wave" for every sample of
one lead at 500 Hz (band 0.5-40 Hz, scaled by its robust SD); the P peaks are the centres of the regions with probability
>= 0.6. LUDB odd ids, lead II: recall 0.964, precision 0.993 (p_chain 0.86 / 0.99). In AF it sees "P" in the f waves —
callers decide AF first. Without onnxruntime or the model file every function returns None (the rules then do without).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[3] / "models" / "p_detector" / "p_cnn_d295_it2.onnx"
FS = 500
THRESHOLD = 0.6
MIN_REGION = 15          # samples (30 ms)


@lru_cache(maxsize=1)
def _session():
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not MODEL.exists():
        return None
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])


def _prep(x: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, resample_poly
    x = np.asarray(x, float)
    if int(round(fs)) != FS:
        x = resample_poly(x, FS, int(round(fs)))
    b, a = butter(2, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band")
    y = filtfilt(b, a, x)
    s = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-6
    return (y / s).astype(np.float32)


def p_peaks(x: np.ndarray, fs: float) -> np.ndarray | None:
    """P-wave peak positions in samples of the INPUT rate ``fs``; None when the detector is unavailable."""
    sess = _session()
    if sess is None or x is None or len(x) < int(2 * fs) or not np.isfinite(x).all():
        return None
    y = _prep(x, fs)
    logit = sess.run(None, {"x": y[None, None, :]})[0][0, 0]
    on = (1.0 / (1.0 + np.exp(-logit))) >= THRESHOLD
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int), [0]))))
    cen = [(a + b) // 2 for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_REGION]
    return np.round(np.asarray(cen, float) * fs / FS).astype(int)


def dissociation_features(p: np.ndarray | None, rpeaks, fs: float) -> dict | None:
    """D296: atrial rate, P-P regularity, atrial / ventricular rate ratio and the spread of the P -> next-QRS interval."""
    rp = np.asarray(rpeaks or [], int)
    if p is None or len(p) < 4 or rp.size < 3:
        return None
    pp = np.diff(p) / fs
    pp = pp[pp > 0.2]
    if pp.size < 3:
        return None
    period = float(np.median(pp))
    reg = float(np.mean(np.abs(pp - period) / period <= 0.15))
    vrate = 60.0 / float(np.median(np.diff(rp) / fs))
    pr = []
    for q in p:
        nxt = rp[rp > q + int(0.04 * fs)]
        if nxt.size:
            pr.append((nxt[0] - q) / fs)
    pr = np.asarray(pr)
    iqr = float(np.subtract(*np.percentile(pr, [75, 25]))) if pr.size >= 4 else None
    return {"p_rate": round(60.0 / period, 1), "p_regular": round(reg, 3), "ratio": round(60.0 / period / vrate, 3),
            "pr_iqr_ms": None if iqr is None else round(iqr * 1000.0, 1), "n_p": int(len(p)), "n_r": int(rp.size)}
