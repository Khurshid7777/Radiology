"""SD-168: the panels' windows put back in time by the beats.

A sheet left turned by half a degree after rectification (or with a horizontal scale a little off) shifts each row of
panels in time against the others: 15-60 ms per row on the bench, while every row of a 12-lead page was recorded at
the same moment. The measurement takes all leads as simultaneous, and the QRS (QT) of the page spreads by the skew
(SD-166). The rhythm strip runs the whole record and serves as the clock: each panel's beats are matched to the
strip's, and one plane over the page (shift = a * y + b * x + c, y = the panel's row, x = its middle) is fitted to
those matches with outliers dropped, so that a lead with small, odd QRS cannot pull its own window aside (SD-167 v1).
The page as a whole is corrected only when that plane moves its windows apart by ``min_span_s`` or more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Alignment:
    shifts_s: dict[str, float] = field(default_factory=dict)   # panel lead -> shift applied (s), the common offset included
    offset_s: float = 0.0                                        # the whole record moved by this (time may not go below 0)
    rr_s: float | None = None
    mad_s: float | None = None
    used: int = 0                                                # panels whose match entered the fit
    why_not: str | None = None                                   # why nothing was moved


def envelope(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Smoothed |derivative|, standardised: where the QRS complexes are, whatever their polarity or size."""
    x = np.asarray(x, float)
    med = np.nanmedian(x) if np.isfinite(x).any() else 0.0
    x = np.nan_to_num(x - med)
    d = np.abs(np.diff(x, prepend=x[:1]))
    w = max(1, int(win_s * fs))
    e = np.convolve(d, np.ones(w) / w, mode="same")
    return (e - e.mean()) / (e.std() + 1e-9)


def beat_period(strip: np.ndarray, fs: float, *, win_s: float, lo_s: float = 0.25, hi_s: float = 2.0,
                min_share: float = 0.3) -> float | None:
    """The strip's RR (s) from its envelope's autocorrelation; None when it does not repeat clearly."""
    e = envelope(strip, fs, win_s)
    n = len(e)
    hi = min(int(hi_s * fs), n - 2)
    lo = int(lo_s * fs)
    if hi <= lo:
        return None
    a = np.array([float(np.dot(e[:n - k], e[k:]) / (n - k)) for k in range(hi + 1)])
    k = lo + int(np.argmax(a[lo:]))
    return k / fs if a[k] > min_share * a[0] else None


def panel_shift(panel: np.ndarray, p_start_s: float, strip: np.ndarray, s_start_s: float, fs: float, *,
                win_s: float, max_shift_s: float) -> tuple[float | None, float]:
    """(shift_s, peak): moving the panel by shift_s matches the strip's envelope best (peak: the correlation there)."""
    ep, er = envelope(panel, fs, win_s), envelope(strip, fs, win_s)
    i0 = int(round((p_start_s - s_start_s) * fs))
    m = int(max_shift_s * fs)
    n = len(ep)
    best: tuple[float | None, float] = (None, -9.0)
    for k in range(-m, m + 1):
        a = i0 + k
        if a < 0 or a + n > len(er):
            continue
        c = float(np.dot(ep, er[a:a + n]) / n)
        if c > best[1]:
            best = (k / fs, c)
    return best


def fit_plane(points: np.ndarray, shifts: np.ndarray, weights: np.ndarray, *, min_n: int, rounds: int = 3,
              floor_s: float = 0.015) -> tuple[np.ndarray | None, float | None, np.ndarray]:
    """Weighted least squares of shift = a*y + b*x + c over (y, x) points, the matches far off the plane dropped
    (3 x median residual, at least ``floor_s``). Returns (coef or None, median abs residual, kept mask)."""
    design = np.column_stack([points, np.ones(len(points))])
    keep = np.ones(len(shifts), bool)
    coef, mad = None, None
    for _ in range(rounds):
        if keep.sum() < min_n:
            return None, None, keep
        coef, *_ = np.linalg.lstsq(design[keep] * weights[keep, None], shifts[keep] * weights[keep], rcond=None)
        res = shifts - design @ coef
        mad = float(np.median(np.abs(res[keep])))
        keep = np.abs(res) <= max(3 * mad, floor_s)
    if keep.sum() < min_n:
        return None, None, keep
    return coef, mad, keep


def align_readings(readings, fs: float, cfg: dict) -> Alignment:
    """Shift the panel traces of ``readings`` (in place) by the page's plane; the rhythm strips move by the common
    offset only. Nothing changes when there is no strip of ``min_strip_s``, too few matches, a poor fit or a small skew."""
    out = Alignment()
    win = float(cfg.get("envelope_s", 0.04))
    strips = [r for r in readings if r.panel.kind == "rhythm" and len(r.trace.samples_mv) / fs >= float(cfg.get("min_strip_s", 8.0))]
    panels = [r for r in readings if r.panel.kind == "panel"]
    min_n = int(cfg.get("min_panels", 6))
    if not strips or len(panels) < min_n:
        out.why_not = "no strip" if not strips else "few panels"
        return out
    ref = strips[0].trace
    strip = ref.array()
    out.rr_s = beat_period(strip, fs, win_s=win)
    if out.rr_s is None:
        out.why_not = "no clear beat"
        return out
    # a shift near half a beat is ambiguous: the neighbouring beat matches as well (a child's fast heart, SD-167 v6)
    max_shift = min(float(cfg.get("max_shift_s", 0.25)), float(cfg.get("rr_share", 0.45)) * out.rr_s)
    pts, sh, wt, who = [], [], [], []
    for r in panels:
        s, peak = panel_shift(r.trace.array(), r.trace.start_s, strip, ref.start_s, fs, win_s=win, max_shift_s=max_shift)
        if s is None or peak < float(cfg.get("min_peak", 0.3)):
            continue
        pts.append((r.panel.baseline, 0.5 * (r.panel.x0 + r.panel.x1)))
        sh.append(s)
        wt.append(peak)
        who.append(r)
    if len(sh) < min_n:
        out.why_not = "few matches"
        return out
    coef, mad, keep = fit_plane(np.array(pts, float), np.array(sh), np.array(wt), min_n=min_n)
    out.used = int(keep.sum())
    out.mad_s = None if mad is None else round(mad, 4)
    if coef is None or mad is None or mad > float(cfg.get("max_mad_s", 0.02)):
        out.why_not = "poor fit"
        return out
    shift = {id(r): float(coef @ [r.panel.baseline, 0.5 * (r.panel.x0 + r.panel.x1), 1.0]) for r in panels}
    if max(shift.values()) - min(shift.values()) < float(cfg.get("min_span_s", 0.03)):
        out.why_not = "small skew"
        return out
    # time may not go below 0: the whole record (the strips with it) moves by one constant instead of clipping a column
    out.offset_s = max(0.0, -min(r.trace.start_s + shift[id(r)] for r in panels))
    for r in panels:
        r.trace.start_s = round(r.trace.start_s + shift[id(r)] + out.offset_s, 4)
        out.shifts_s[r.panel.lead] = round(shift[id(r)] + out.offset_s, 4)
    for r in readings:
        if r.panel.kind == "rhythm" and out.offset_s:
            r.trace.start_s = round(r.trace.start_s + out.offset_s, 4)
    return out
