"""Beat coherence of a digitised lead (SD-117): does the lead repeat itself from beat to beat?

A well read lead of a regular rhythm shows the same complex again and again; a lead whose trace
wandered onto a neighbour row, lost the baseline or followed noise does not. The check needs no
reference signal: the beats are found from all leads at once (a weak minority cannot move them), and
each lead is compared with its own median beat. It judges how the page was read, not the heart: an
irregular rhythm lowers it too, which is why it can only ask the doctor to look, never reject.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ecgscan.contract import Trace


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def beat_times(traces: list[Trace], fs: float, *, min_rr_s: float = 0.25, smooth_s: float = 0.04,
               height: float = 0.35) -> np.ndarray:
    """Beat positions (s) from the steepness of every trace together, each scaled by its own typical steep value."""
    if not traces:
        return np.zeros(0)
    end = max(t.start_s + len(t.array()) / fs for t in traces)
    n = int(np.ceil(end * fs)) + 1
    acc, cnt = np.zeros(n), np.zeros(n)
    for t in traces:
        a = t.array().astype(float)
        if a.size < int(0.5 * fs):
            continue
        ok = np.isfinite(a)
        d = np.abs(np.gradient(np.where(ok, a, 0.0)))
        d[~ok] = 0.0
        e = _smooth(d, max(1, int(round(smooth_s * fs))))
        scale = np.percentile(e[ok], 98) if ok.any() else 0.0
        if scale <= 0:
            continue
        i0 = int(round(t.start_s * fs))
        acc[i0:i0 + e.size] += np.minimum(e / scale, 1.5)[: n - i0]
        cnt[i0:i0 + e.size] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        e_all = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    if e_all.max() <= 0:
        return np.zeros(0)
    pk, _ = find_peaks(e_all, height=height * e_all.max(), distance=max(1, int(min_rr_s * fs)))
    return pk / fs


def lead_coherence(trace: Trace, beats: np.ndarray, fs: float, *, pre_s: float = 0.15, post_s: float = 0.30,
                   min_beats: int = 3) -> float | None:
    """Median correlation of the lead's beat windows with their median template (None: too few beats here)."""
    a = trace.array().astype(float)
    i0 = int(round(trace.start_s * fs))
    p, q = int(round(pre_s * fs)), int(round(post_s * fs))
    wins = []
    for b in beats:
        c = int(round(b * fs)) - i0
        if c - p < 0 or c + q > a.size:
            continue
        w = a[c - p:c + q]
        if not np.isfinite(w).all():
            continue
        w = w - w.mean()
        s = w.std()
        if s < 1e-6:
            wins.append(None)           # a flat window is a beat that is not there
            continue
        wins.append(w / s)
    real = [w for w in wins if w is not None]
    if len(wins) < min_beats or len(real) < 2:
        return None if len(wins) < min_beats else 0.0
    tpl = np.median(np.stack(real), axis=0)
    ts = tpl.std()
    if ts < 1e-6:
        return 0.0
    tpl = (tpl - tpl.mean()) / ts
    cs = [float(np.mean(w * tpl)) if w is not None else 0.0 for w in wins]
    return float(np.median(cs))


def coherence_by_lead(panels: dict[str, Trace], rhythm: list[Trace], fs: float) -> dict[str, float | None]:
    """Coherence of every panel lead (the rhythm strips take part in finding the beats, not in the answer)."""
    traces = [*panels.values(), *rhythm]
    beats = beat_times(traces, fs)
    return {lead: lead_coherence(tr, beats, fs) for lead, tr in panels.items()}
