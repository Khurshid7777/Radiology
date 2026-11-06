"""Atrial fibrillation features and the second-path model (D205).

The rule (layer 4) reads AF from an irregular RR with absent / fibrillatory P waves. It misses AF whose
ventricular response is nearly regular or whose f-waves pass for P, and then the report could say
"sinus rhythm" or "supraventricular ectopy". This module computes a small, named feature vector from
the measurements and the median-beat-free signal, and a model trained on Chapman/Ningbo (never on the
PTB-XL bench it is checked on) gives P(AF). The model never sets the AF finding itself: see
``pipeline._af_second_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AF_FEATURES = (
    "n_rr", "hr", "rr_cv", "rr_mad_cv", "rr_big", "rr_big_trim", "rr_rmssd_norm", "rr_sd_ratio",
    "rr_turning", "rr_max_min", "pc_corr", "pc_keep", "pc_corr_v1", "pc_keep_v1", "fwave_index",
    "p_wave_ratio", "p_onset_ratio", "pr_std_ms", "atrial_rate", "qrs_ms",
)
# Plain JSON (scaler + logistic coefficients), not a pickle: loading it runs no code.
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "cardiolens" / "af_second_path.json"
_MISSING = -1.0


def _rr_features(rpeaks, fs: float) -> dict[str, float]:
    rp = np.asarray(rpeaks, dtype=float)
    if rp.size < 6:
        return {}
    rr = np.diff(rp) / fs * 1000.0
    mean = float(rr.mean())
    d = np.diff(rr)
    order = np.argsort(np.abs(rr - np.median(rr)))
    trim = rr[np.sort(order[:max(4, rr.size - 2)])]           # without the two most deviant RR (ectopy)
    sd1 = float(np.std(d) / np.sqrt(2))
    sd2 = float(np.sqrt(max(1e-9, 2 * np.var(rr) - sd1 ** 2)))
    turning = float(np.mean((d[1:] * d[:-1]) < 0)) if d.size > 1 else 0.0
    return {
        "n_rr": float(rr.size),
        "rr_big": float(np.mean(np.abs(d) > 0.08 * mean)),
        "rr_big_trim": float(np.mean(np.abs(np.diff(trim)) > 0.08 * trim.mean())),
        "rr_rmssd_norm": float(np.sqrt(np.mean(d ** 2)) / mean),
        "rr_sd_ratio": sd1 / sd2,
        "rr_turning": turning,
        "rr_max_min": float(rr.max() / rr.min()),
    }


def af_features(sig, m) -> dict[str, float]:
    """Named AF features of one record (missing values as -1)."""
    from cardiolens.layer4_findings.p_consistency import p_window_consistency

    f = {k: _MISSING for k in AF_FEATURES}
    f.update(_rr_features(m.r_peaks_samples or [], sig.sampling_rate_hz))
    for lead, suffix in (("II", ""), ("V1", "_v1")):
        pc = p_window_consistency(sig, m.r_peaks_samples or [], lead=lead)
        if pc is not None:
            f["pc_corr" + suffix], f["pc_keep" + suffix] = pc
    for key, val in (("hr", m.heart_rate_bpm), ("rr_cv", m.rr_cv), ("rr_mad_cv", m.rr_mad_cv),
                     ("fwave_index", m.fwave_index), ("p_wave_ratio", m.p_wave_ratio),
                     ("p_onset_ratio", m.p_onset_ratio), ("pr_std_ms", m.pr_std_ms),
                     ("atrial_rate", m.atrial_rate_bpm), ("qrs_ms", m.intervals.qrs_ms)):
        if val is not None:
            f[key] = float(val)
    return f


def vector(f: dict[str, float]) -> np.ndarray:
    return np.array([[f.get(k, _MISSING) for k in AF_FEATURES]], dtype=float)


_MODEL: dict | None = None


def load_af_model(path: Path = MODEL_PATH) -> dict | None:
    """The model dict (features, mean, scale, coef, intercept, threshold, ...) or None when absent."""
    global _MODEL
    if _MODEL is None and path.exists():
        import json

        _MODEL = json.loads(path.read_text(encoding="utf-8"))
    return _MODEL


def af_probability(f: dict[str, float], model: dict) -> float:
    x = (vector(f)[0] - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = float(np.dot(x, np.asarray(model["coef"])) + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-z)))
