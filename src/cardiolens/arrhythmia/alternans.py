"""Electrical alternans (D282, wave 5 A6): QRS amplitude higher / lower every other beat.

Total electrical alternans with sinus tachycardia is a sign of a large pericardial effusion / tamponade (ESC 2015
pericardial guidelines; Surawicz & Knilans ch. 13). Per lead, the QRS peak-to-peak of each beat (R +- 60 ms, 1-40 Hz)
is taken; the lead alternates when, over at least 8 beats, successive amplitude changes flip sign in at least 80 % of
pairs and the median change is at least 10 % of the median amplitude (and at least 100 µV). Alternans is called when
at least two leads alternate and the RR intervals do not alternate (that would be bigeminy, not alternans).
Thresholds: DECISIONS D282 (synthetic + false alarms on PTB-XL).
"""
from __future__ import annotations

import numpy as np

ALT_MIN_BEATS = 8
ALT_FLIP_MIN = 0.80
ALT_REL_MIN = 0.10
ALT_ABS_MIN_UV = 100.0
ALT_LEADS_MIN = 2
ALT_RR_FLIP_MAX = 0.60      # RR alternating in more pairs -> bigeminy-like, not alternans
ALT_RR_REL_MIN = 0.05


def _flip_share(v: np.ndarray, rel_min: float = 0.0) -> tuple[float, float]:
    d = np.diff(v)
    if d.size < 2:
        return 0.0, 0.0
    flips = np.sign(d[1:]) != np.sign(d[:-1])
    big = np.abs(d) >= rel_min * np.median(np.abs(v)) if rel_min else np.ones_like(d, bool)
    return float(np.mean(flips & big[1:] & big[:-1])), float(np.median(np.abs(d)))


def lead_alternans(x: np.ndarray, rpeaks: np.ndarray, fs: float) -> dict | None:
    from scipy.signal import butter, filtfilt

    half = int(0.06 * fs)
    rp = rpeaks[(rpeaks - half >= 0) & (rpeaks + half < x.size)]
    if rp.size < ALT_MIN_BEATS:
        return None
    nyq = fs / 2.0
    b, a = butter(2, [1.0 / nyq, min(40.0, 0.45 * fs) / nyq], btype="band")
    y = filtfilt(b, a, np.asarray(x, float))
    amp = np.array([np.ptp(y[r - half:r + half]) for r in rp])
    med = float(np.median(amp))
    if med <= 0:
        return None
    flip, dmed = _flip_share(amp)
    return {"flip": round(flip, 3), "rel": round(dmed / med, 3), "dmed_uv": round(dmed, 1), "n": int(rp.size)}


def detect_alternans(signal: np.ndarray, lead_names: list[str], rpeaks, fs: float) -> tuple[bool, dict]:
    """(alternans?, details). ``signal`` leads x samples in µV."""
    rp = np.asarray(rpeaks or [], int)
    out: dict = {"leads": {}}
    if rp.size < ALT_MIN_BEATS + 1:
        return False, out
    rr = np.diff(rp).astype(float)
    rr_flip, _ = _flip_share(rr, ALT_RR_REL_MIN)
    out["rr_flip"] = round(rr_flip, 3)
    hits = []
    for name, row in zip(lead_names, np.atleast_2d(signal)):
        m = lead_alternans(row, rp, fs)
        if m is None:
            continue
        out["leads"][name] = m
        if m["flip"] >= ALT_FLIP_MIN and m["rel"] >= ALT_REL_MIN and m["dmed_uv"] >= ALT_ABS_MIN_UV:
            hits.append(name)
    out["hits"] = hits
    return bool(len(hits) >= ALT_LEADS_MIN and rr_flip <= ALT_RR_FLIP_MAX), out
