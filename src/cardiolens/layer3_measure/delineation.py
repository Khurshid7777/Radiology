"""Robust delineation helpers layered over neurokit's dwt output (TZ §6).

Real 12-lead recordings (PhysioNet ecg-arrhythmia) exposed two weaknesses of the
pure dwt delineation: QRS width is under-estimated (bundle branch blocks read as
narrow) and the P onset lands late (PR under-estimated). These helpers derive QRS
bounds from the multi-lead spatial-velocity envelope and refine P onsets by
amplitude. Where a beat cannot be delineated the value is left NaN — never
fabricated (TZ §0).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

# Envelope threshold as a fraction of the per-beat envelope maximum. Calibrated on
# PhysioNet ecg-arrhythmia: at 0.12 the slurred terminal forces of a bundle branch
# block fall below threshold (RBBB read as 62-74 ms); at 0.05 RBBB/LBBB measure
# 112-190 ms while normal QRS stays <= 102 ms.
QRS_ENVELOPE_FRACTION = 0.05
# A width beyond this is not a QRS (envelope ran into the T wave) -> NaN.
QRS_MAX_MS = 220.0
# Search half-window around each R peak, and how long the envelope must stay
# below threshold to count as "outside the QRS".
QRS_SEARCH_MS = 150.0
QRS_QUIET_MS = 8.0
# The threshold is never below this multiple of the record's envelope floor (median
# envelope over the middle third of each RR). Fibrillatory waves or noise on a
# low-voltage QRS lift the floor above 0.05 of the QRS peak: the envelope is never
# "quiet" and the bounds slide outward (PTB-XL 3166, AF at 150/min: floor 0.069,
# QRS 122 ms vs 86 ms by GE 12SL -> false wide-complex tachycardia alert). Clean
# records sit at 0.01-0.03 and keep the 0.05 threshold.
QRS_FLOOR_FACTOR = 1.5
# Terminal hump (D90): an envelope dip after the main QRS deflection can be longer than
# the quiet window while a terminal R'/S (RBBB) follows. When the envelope rises again
# above this fraction of the beat peak within QRS_HUMP_GAP_MS of the offset found, the
# offset walk continues past that hump. 0 disables.
QRS_HUMP_FRACTION = 0.15
QRS_HUMP_GAP_MS = 40.0

# P-onset refinement: search window before QRS onset, onset criterion and cap.
P_SEARCH_BEFORE_MS = 320.0
P_SEARCH_UNTIL_MS = 40.0
# Onset = where the smoothed deflection drops below this fraction of the P peak
# (with an absolute floor above the noise of a 0.5-40 Hz band-passed ECG), at most
# P_ONSET_MAX_MS before the peak; if never met, the onset is the start of the
# supra-threshold run (NOT a fixed offset, which inflated PR by 60-80 ms).
P_ONSET_FRACTION = 0.15
P_ONSET_MIN_LEVEL_UV = 12.0
P_ONSET_MAX_MS = 80.0
P_SMOOTH_MS = 10.0
P_BASELINE_MS = 30.0
# A P-wave candidate run above the floor must be this wide (ms): narrower is
# noise, wider is a T wave.
P_RUN_WIDTH_MS = (20.0, 160.0)
# Amplitude floor for P candidates as a fraction of the R reference (lower than
# the AF gate: we only need a plausible P, not a certainly organised one).
P_ONSET_MIN_FRACTION = 0.05
P_ONSET_MIN_UV = 30.0

# Terminal-QRS window (last N ms before the QRS offset) for BBB typing.
TERMINAL_MS = 40.0


def spatial_velocity(signal: np.ndarray, fs: float) -> np.ndarray:
    """RMS across leads of the first derivative, smoothed over ~10 ms.

    The spatial-velocity envelope peaks sharply during the QRS in every lead at
    once, which makes the QRS bounds far less sensitive to single-lead morphology
    (notches, R', deep S) than a one-lead delineation.
    """
    x = signal.astype(np.float64)
    d = np.diff(x, axis=1, prepend=x[:, :1])
    env = np.sqrt(np.mean(d * d, axis=0))
    k = max(1, int(round(0.010 * fs)))
    return np.convolve(env, np.ones(k) / k, mode="same")


def envelope_floor(env: np.ndarray, rpeaks: np.ndarray) -> float:
    """Background level of the envelope: median over the middle third of each RR,
    away from the QRS (and mostly from the T wave). 0.0 with fewer than two beats."""
    rs = np.sort(np.asarray(rpeaks, dtype=int))
    mids = [env[x + (y - x) // 3: y - (y - x) // 3] for x, y in pairwise(rs) if y - x >= 6]
    return float(np.median(np.concatenate(mids))) if mids else 0.0


def qrs_bounds_envelope(
    signal: np.ndarray, rpeaks: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat QRS onset/offset (sample indices, NaN when not found)."""
    n_beats = len(rpeaks)
    on = np.full(n_beats, np.nan)
    off = np.full(n_beats, np.nan)
    if n_beats == 0:
        return on, off
    env = spatial_velocity(signal, fs)
    n = env.size
    half = int(QRS_SEARCH_MS / 1000.0 * fs)
    quiet = max(1, int(round(QRS_QUIET_MS / 1000.0 * fs)))
    gap = int(round(QRS_HUMP_GAP_MS / 1000.0 * fs))
    floor = envelope_floor(env, rpeaks)
    for i, r in enumerate(rpeaks):
        a, b = max(0, int(r) - half), min(n, int(r) + half + 1)
        if b - a < 3:
            continue
        seg = env[a:b]
        thr = max(QRS_ENVELOPE_FRACTION * float(seg.max()), QRS_FLOOR_FACTOR * floor)
        below = seg < thr
        ri = int(r) - a
        # Onset: walk left from R until the envelope has been quiet for `quiet`.
        run, j = 0, ri
        while j > 0:
            j -= 1
            run = run + 1 if below[j] else 0
            if run >= quiet:
                on[i] = a + j + quiet
                break
        # Offset: walk right from R likewise, then past a terminal hump.
        j, peak = ri, float(seg.max())
        while True:
            run, end = 0, np.nan
            while j < len(seg) - 1:
                j += 1
                run = run + 1 if below[j] else 0
                if run >= quiet:
                    end = j - quiet + 1
                    break
            if not np.isfinite(end):
                break
            off[i] = a + end
            g = seg[int(end): int(end) + gap]
            if QRS_HUMP_FRACTION <= 0 or g.size == 0 or g.max() < max(QRS_HUMP_FRACTION * peak, 2.0 * thr):
                break
            j = int(end) + int(np.argmax(g))
        if np.isfinite(on[i]) and np.isfinite(off[i]) and (off[i] - on[i]) > QRS_MAX_MS / 1000.0 * fs:
            on[i] = off[i] = np.nan
    return on, off


def refine_p_onsets(
    row: np.ndarray,
    qrs_on: np.ndarray,
    t_off: np.ndarray | None,
    fs: float,
    min_amp_uv: float,
) -> np.ndarray:
    """Amplitude-based P onset per beat on one lead (NaN when no P is found).

    Searches [qrs_on − 320 ms, qrs_on − 40 ms] (not before the previous T offset)
    with the PR-segment median as baseline. Candidate deflections are contiguous
    runs above ``min_amp_uv`` that are 20–160 ms wide (a T wave is wider, noise is
    narrower); the run **closest to the QRS** is the P wave — taking the largest
    run instead picks up the previous T/U wave in bradycardia. The onset is where
    the deflection drops below 10 % of the P amplitude walking back from its peak.
    """
    n_beats = len(qrs_on)
    p_on = np.full(n_beats, np.nan)
    before = int(P_SEARCH_BEFORE_MS / 1000.0 * fs)
    until = int(P_SEARCH_UNTIL_MS / 1000.0 * fs)
    base_w = int(P_BASELINE_MS / 1000.0 * fs)
    need = max(1, int(round(0.006 * fs)))
    cap = int(P_ONSET_MAX_MS / 1000.0 * fs)
    w_min, w_max = int(P_RUN_WIDTH_MS[0] / 1000.0 * fs), int(P_RUN_WIDTH_MS[1] / 1000.0 * fs)
    smooth = max(1, int(round(P_SMOOTH_MS / 1000.0 * fs)))
    for i in range(n_beats):
        if not np.isfinite(qrs_on[i]):
            continue
        on = int(qrs_on[i])
        lo, hi = on - before, on - until
        if t_off is not None and i > 0 and i - 1 < len(t_off) and np.isfinite(t_off[i - 1]):
            lo = max(lo, int(t_off[i - 1]) + int(0.02 * fs))
        lo = max(lo, 0)
        if hi - lo < int(0.04 * fs) or on - base_w < 0:
            continue
        base = float(np.median(row[on - base_w:on]))
        seg = row[lo:hi] - base
        mask = np.abs(seg) >= min_amp_uv
        if not mask.any():
            continue
        # Contiguous runs above the floor, scanned from the QRS backwards.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
        runs = list(zip(edges[::2], edges[1::2]))
        k = None
        for start, stop in reversed(runs):
            if w_min <= stop - start <= w_max:
                k = start + int(np.argmax(np.abs(seg[start:stop])))
                break
        # `start` of the chosen run doubles as the onset fallback below.
        if k is None:
            continue
        amp = float(abs(seg[k]))
        sm = np.convolve(np.abs(seg), np.ones(smooth) / smooth, mode="same")
        level = max(P_ONSET_FRACTION * amp, P_ONSET_MIN_LEVEL_UV)
        limit = max(0, k - cap)
        onset, run, j = max(limit, start), 0, k
        while j > limit:
            j -= 1
            if sm[j] < level:
                run += 1
                if run >= need:
                    onset = j + run
                    break
            else:
                run = 0
        p_on[i] = lo + onset
    return p_on


TERMINAL_SLOPE_FRACTION = 0.15   # lead-specific QRS end = last point this steep (of the lead's QRS slope)


def terminal_qrs_sign(
    row: np.ndarray, rpeaks: np.ndarray, qrs_off: np.ndarray, fs: float,
    qrs_on: np.ndarray | None = None,
) -> bool | None:
    """Sign of the terminal QRS forces on one lead (True = positive).

    Averages the last ``TERMINAL_MS`` of the lead's OWN QRS: the multi-lead
    envelope offset can be later than this lead's last deflection, and the window
    then swallows the discordant ST segment (LBBB read as RBBB on 26 of 37 PTB-XL
    misses, D41). The lead's QRS end is the last sample before the envelope offset
    whose slope is still ``TERMINAL_SLOPE_FRACTION`` of the lead's QRS slope.
    Baseline = PR segment when ``qrs_on`` is given, else the lead median.
    """
    if len(rpeaks) == 0:
        return None
    w = int(TERMINAL_MS / 1000.0 * fs)
    k = max(1, int(0.005 * fs))
    slope = np.abs(np.convolve(np.diff(row, prepend=row[0]), np.ones(k) / k, mode="same"))
    vals = []
    for i, r in enumerate(rpeaks):
        r = int(r)
        if qrs_on is not None and i < len(qrs_on) and np.isfinite(qrs_on[i]) and qrs_on[i] > int(0.05 * fs):
            on = int(qrs_on[i])
            baseline = float(np.median(row[on - int(0.05 * fs): max(on - int(0.01 * fs), on - int(0.05 * fs) + 1)]))
        else:
            baseline = float(np.median(row))
        if i < len(qrs_off) and np.isfinite(qrs_off[i]) and qrs_off[i] > r:
            off = int(qrs_off[i])
            seg = slope[max(0, r - int(0.06 * fs)): off + 1]
            if seg.size:
                steep = np.flatnonzero(slope[r: off + 1] >= TERMINAL_SLOPE_FRACTION * float(seg.max()))
                end = r + int(steep[-1]) + 1 if steep.size else off
            else:
                end = off
            a, b = end - w, end
        else:
            a, b = r + int(0.02 * fs), r + int(0.06 * fs)
        if 0 <= a < b <= len(row):
            vals.append(float(np.mean(row[a:b]) - baseline))
    if not vals:
        return None
    return bool(np.median(vals) > 0)


# --- T offset by the tangent method, per lead -----------------------------------
# The dwt T offset on a single lead landed early on real records (measured QT of
# 196-290 ms on long-QT cases). Guideline practice: measure QT in the lead with the
# longest QT, T offset = intersection of the steepest descending tangent of the T
# wave with the baseline.
T_WINDOW_START_MS = 80.0      # after the QRS offset
T_WINDOW_MAX_MS = 520.0       # ... and no further than this (or 0.55 * RR)
T_WINDOW_RR_FRACTION = 0.55
T_MIN_AMP_UV = 80.0           # smaller T waves are not delineated on that lead
T_SLOPE_SMOOTH_MS = 10.0
T_MIN_VALID_FRACTION = 0.5    # a lead needs an offset on this share of beats
# Across-lead aggregate of the per-lead median QT. The textbook "lead with the
# longest QT" is the max of 12 noisy estimates and over-read on real data (664
# PhysioNet records: normal QTc median 441-457, spec 0.44-0.57). The 75th
# percentile keeps the long-QT sensitivity (0.56-0.83 by split) with normal QTc
# back at ~405 (Fridericia); the median under-reads (sens 0.44-0.58).
T_LEAD_AGGREGATE_PERCENTILE = 75.0


def t_offset_tangent(
    row: np.ndarray, qrs_on: float, qrs_off: float, rr_s: float, fs: float
) -> float | None:
    """T offset (sample index) on one lead for one beat, tangent method."""
    if not (np.isfinite(qrs_on) and np.isfinite(qrs_off)):
        return None
    on, off = int(qrs_on), int(qrs_off)
    b_lo, b_hi = int(0.050 * fs), int(0.010 * fs)
    lo, hi = max(0, on - b_lo), max(1, on - b_hi)
    if hi <= lo:
        return None
    baseline = float(np.median(row[lo:hi]))
    w0 = off + int(T_WINDOW_START_MS / 1000.0 * fs)
    w1 = off + int(min(T_WINDOW_MAX_MS / 1000.0, T_WINDOW_RR_FRACTION * rr_s) * fs)
    if w1 - w0 < int(0.08 * fs) or w1 > len(row):
        return None
    seg = row[w0:w1].astype(np.float64) - baseline
    k = int(np.argmax(np.abs(seg)))
    amp = float(seg[k])
    if abs(amp) < T_MIN_AMP_UV:
        return None
    # Steepest slope back toward baseline after the peak (10 ms smoothed derivative).
    sm = max(1, int(round(T_SLOPE_SMOOTH_MS / 1000.0 * fs)))
    d = np.convolve(np.diff(seg), np.ones(sm) / sm, mode="same")
    tail = d[k:]
    if tail.size < 2:
        return None
    j = k + int(np.argmin(tail) if amp > 0 else np.argmax(tail))
    slope = float(d[j])
    if (amp > 0 and slope >= 0) or (amp < 0 and slope <= 0):
        return None
    # Tangent through (j, seg[j]) with that slope meets the baseline at x where seg=0.
    x = j - seg[j] / slope
    if not (k < x <= len(seg) + int(0.02 * fs)):
        return None
    return float(w0 + x)


def qt_multilead(
    signal: np.ndarray,
    lead_names: list[str],
    rpeaks: np.ndarray,
    qrs_on: np.ndarray,
    qrs_off: np.ndarray,
    fs: float,
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[float | None, str | None, np.ndarray | None]:
    """QT (ms) aggregated across leads, plus the reference lead's per-beat T offsets.

    Returns (qt_ms, lead, t_offsets) or (None, None, None). A lead counts when it
    yields a tangent offset on >= T_MIN_VALID_FRACTION of beats; the record QT is
    the T_LEAD_AGGREGATE_PERCENTILE of the per-lead medians and the reported lead
    is the one closest to that value.
    """
    n = len(rpeaks)
    if n < 2:
        return None, None, None
    rr_s = float(np.median(np.diff(rpeaks))) / fs
    per_lead: list[tuple[float, str, np.ndarray]] = []
    for li, name in enumerate(lead_names):
        offs = np.full(n, np.nan)
        for i in range(n):
            t = t_offset_tangent(signal[li], qrs_on[i], qrs_off[i], rr_s, fs)
            if t is not None:
                offs[i] = t
        valid = np.isfinite(offs) & np.isfinite(qrs_on)
        if masks is not None and name in masks:
            valid &= masks[name]
            n_ref = max(1, int(masks[name].sum()))
        else:
            n_ref = n
        if valid.sum() >= max(2, T_MIN_VALID_FRACTION * n_ref):
            qt = float(np.median(offs[valid] - qrs_on[valid]) / fs * 1000.0)
            per_lead.append((qt, name, offs))
    if not per_lead:
        return None, None, None
    target = float(np.percentile([q for q, _, _ in per_lead], T_LEAD_AGGREGATE_PERCENTILE))
    _, name, offs = min(per_lead, key=lambda t: abs(t[0] - target))
    return round(target, 1), name, offs
