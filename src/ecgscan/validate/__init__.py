"""Physiological consistency and per-lead quality (TZ §5.8).

Limb-lead identities hold for any heart (they follow from how the leads are
derived): II = I + III, aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2, and
therefore aVR + aVL + aVF = 0. Each is checked only where all of its leads were
recorded *at the same time* (a 3x4 page has I/II/III together in 0-2.5 s,
aVR/aVL/aVF together in 2.5-5 s, and the rhythm strip overlaps every panel of its
lead). A violation means a digitisation error or swapped electrodes; either way it
lowers confidence and is flagged — it is never "corrected".

Every signal is compared after removing its own median (the absolute baseline
on paper is unknown to a digitiser).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecgscan.contract import Trace

# name -> (target, {lead: coefficient}) with target = sum(coef * lead)
IDENTITIES = {
    "einthoven": ("II", {"I": 1.0, "III": 1.0}),
    "avr": ("aVR", {"I": -0.5, "II": -0.5}),
    "avl": ("aVL", {"I": 1.0, "II": -0.5}),
    "avf": ("aVF", {"II": 1.0, "I": -0.5}),
    "goldberger_sum": ("aVR", {"aVL": -1.0, "aVF": -1.0}),
}


@dataclass
class Consistency:
    score: float | None                     # 1 = identities hold exactly, None = nothing overlapped
    checks: dict[str, float] = field(default_factory=dict)   # identity -> residual / signal RMS


def _on_axis(tr: Trace, fs: float, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    a = tr.array()
    i0 = int(round(tr.start_s * fs))
    seg = a[: max(0, n - i0)]
    out[i0:i0 + len(seg)] = seg
    return out


def limb_consistency(traces: list[tuple[str, Trace]], fs: float, *, min_overlap_s: float = 1.0) -> Consistency:
    if not traces:
        return Consistency(None)
    n = int(round(max(t.start_s + len(t.samples_mv) / fs for _, t in traces) * fs))
    # Several traces of one lead (a panel and its rhythm strip) -> one axis, panel first.
    axis: dict[str, np.ndarray] = {}
    for lead, tr in traces:
        a = _on_axis(tr, fs, n)
        axis[lead] = a if lead not in axis else np.where(np.isfinite(axis[lead]), axis[lead], a)
    checks = {}
    for name, (target, terms) in IDENTITIES.items():
        leads = [target, *terms]
        if not all(ld in axis for ld in leads):
            continue
        ok = np.ones(n, dtype=bool)
        for ld in leads:
            ok &= np.isfinite(axis[ld])
        if ok.sum() < min_overlap_s * fs:
            continue
        cen = {ld: axis[ld][ok] - np.median(axis[ld][ok]) for ld in leads}
        pred = sum(coef * cen[ld] for ld, coef in terms.items())
        scale = max(float(np.sqrt(np.mean(cen[target] ** 2))), float(np.sqrt(np.mean(pred ** 2))), 0.05)
        checks[name] = round(float(np.sqrt(np.mean((cen[target] - pred) ** 2)) / scale), 4)
    if not checks:
        return Consistency(None)
    worst = max(checks.values())
    return Consistency(round(float(np.clip(1.0 - worst, 0.0, 1.0)), 4), checks)


def rhythm_agreement(panel: Trace, rhythm: Trace, fs: float) -> float | None:
    """Correlation of a lead's panel with its rhythm strip over the panel's time window."""
    n = int(round(max(panel.start_s + len(panel.samples_mv) / fs, rhythm.start_s + len(rhythm.samples_mv) / fs) * fs))
    a, b = _on_axis(panel, fs, n), _on_axis(rhythm, fs, n)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 0.5 * fs or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def lead_quality(missing_fraction: float, ambiguity: float, edge_contact: float, *,
                 weights: dict | None = None) -> float:
    """0..1 from what the tracer saw: coverage, competing ink, contact with the band edge."""
    w = weights or {"ambiguity": 1.0, "edge": 0.5}
    q = (1.0 - missing_fraction) * (1.0 - w["ambiguity"] * min(1.0, ambiguity)) * (1.0 - w["edge"] * min(1.0, edge_contact))
    return float(np.clip(q, 0.0, 1.0))


def _qrs_envelope(x: np.ndarray, fs: float) -> np.ndarray:
    """Smoothed |slope|: peaks at the QRS complexes whatever the lead's polarity."""
    from scipy.ndimage import uniform_filter1d

    d = np.abs(np.diff(np.nan_to_num(x - np.nanmedian(x)), prepend=0.0))
    return uniform_filter1d(d, size=max(3, int(0.04 * fs)))


def _envelope_on_axis(a: np.ndarray, fs: float) -> np.ndarray:
    """QRS envelope over the trace's own finite stretch, NaN elsewhere."""
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    if ok.sum() > 3:
        idx = np.flatnonzero(ok)
        seg = slice(idx[0], idx[-1] + 1)
        x = a[seg].copy()
        x[~np.isfinite(x)] = np.nanmedian(x)
        out[seg] = _qrs_envelope(x, fs)
        out[~ok] = np.nan
    return out


def column_coherence(traces: list[tuple[str, Trace]], fs: float, *, min_len_s: float = 1.0) -> dict[str, float]:
    """Per trace: correlation of its QRS envelope with the median envelope of the *other*
    traces recorded at the same moments (the leads of a 3x4 or 6x2 column, all rows of a
    12x1 strip, the panels under the rhythm strip).

    All leads of one moment share the QRS timing, so a trace that skipped or invented a
    beat, or followed a neighbour, has a low value. Keys: ``lead@start_s`` (``r`` = strip)."""
    if not traces:
        return {}
    n = int(round(max(t.start_s + len(t.samples_mv) / fs for _, t in traces) * fs))
    keys = [f"{ld}@{t.start_s:g}{'r' if len(t.samples_mv) / fs > 5.5 else ''}" for ld, t in traces]
    envs = np.vstack([_envelope_on_axis(_on_axis(t, fs, n), fs) for _, t in traces])
    out: dict[str, float] = {}
    for i, key in enumerate(keys):
        others = np.delete(envs, i, axis=0)
        cnt = np.isfinite(others).sum(axis=0)
        ok = np.isfinite(envs[i]) & (cnt >= 2)
        if ok.sum() < min_len_s * fs:
            continue
        ref = np.nanmedian(others[:, ok], axis=0)
        own = envs[i, ok]
        if own.std() == 0 or ref.std() == 0:
            continue
        out[key] = round(float(np.corrcoef(own, ref)[0, 1]), 4)
    return out
