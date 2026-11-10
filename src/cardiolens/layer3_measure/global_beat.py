"""Global (all-lead) wave boundaries from the 12-lead median beat (D30).

A single lead's T offset is noisy: against GE 12SL global measurements on PTB-XL
development folds the rhythm-lead QT had MAE 46 ms (28% of records off by > 60 ms).
Commercial interpreters measure QT on the superimposed leads instead. Here every
lead's median beat is aligned on the shared R peaks, referenced to its own
isoelectric PR segment, and combined:

* QRS onset/offset — summed slope energy crossing 5% of its QRS peak;
* T end — tangent through the steepest descent of the RMS curve after the T peak,
  intersected with the isoelectric noise level, capped at the point where the RMS
  falls within 5% of the T peak; the search stops 100 ms before the next beat
  (an unbounded window ran into the next QRS at fast rates);
* P onset — the RMS curve falling to 10% of the P peak above the noise level.

Offsets against a reference are applied by the caller (``DelineationCorrection``).
"""

from __future__ import annotations

import numpy as np

_PRE_S, _POST_S = 0.35, 0.65


def median_beats(signal: np.ndarray, rpeaks, fs: float) -> tuple[np.ndarray | None, int]:
    """Per-lead median beat (leads x samples) around the R peaks; R at index ``a``."""
    a, b = int(_PRE_S * fs), int(_POST_S * fs)
    rp = [int(r) for r in rpeaks if r - a >= 0 and r + b < signal.shape[1]]
    if len(rp) < 3:
        return None, a
    beats = np.stack([signal[:, r - a:r + b] for r in rp]).astype(float)
    return np.median(beats, axis=0), a


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def global_fiducials(X: np.ndarray, a: int, fs: float, rr_s: float | None,
                     p_fraction: float = 0.10) -> dict[str, int | None]:
    """Sample indices (in the median beat) of P onset, QRS onset/offset and T end."""
    n = X.shape[1]
    w20 = max(2, int(0.02 * fs))
    lo, hi = max(0, a - int(0.30 * fs)), a - int(0.03 * fs)
    base = []
    for x in X:                       # isoelectric level: flattest 20 ms in the PR segment
        best, val = np.inf, 0.0
        for s in range(lo, max(lo + 1, hi - w20)):
            seg = x[s:s + w20]
            e = float(np.ptp(seg))
            if e < best:
                best, val = e, float(np.mean(seg))
        base.append(val)
    xb = X - np.asarray(base)[:, None]
    k = max(1, int(0.01 * fs))
    rms = _smooth(np.sqrt(np.mean(xb ** 2, axis=0)), k)
    energy = _smooth(np.sum(np.abs(np.diff(xb, axis=1)), axis=0), k)
    win = energy[max(0, a - int(0.1 * fs)):a + int(0.1 * fs)]
    thr = 0.05 * float(win.max()) if win.size else 0.0
    on = a
    while on > a - int(0.12 * fs) and on > 0 and energy[on - 1] >= thr:
        on -= 1
    off = a
    while off < min(a + int(0.2 * fs), len(energy) - 1) and energy[off] >= thr:
        off += 1
    noise = float(np.median(rms[lo:hi])) if hi > lo else 0.0

    out: dict[str, int | None] = {"p_on": None, "qrs_on": on, "qrs_off": off, "t_end": None}
    t0, t1 = off + int(0.06 * fs), min(n - 1, off + int(0.55 * fs))
    if rr_s:
        t1 = max(min(t1, a + int((rr_s - 0.10) * fs)), t0 + int(0.10 * fs))
    t1 = min(t1, n - 1)
    if t1 - t0 >= 5:
        tp = t0 + int(np.argmax(rms[t0:t1]))
        # Latest plausible end: the RMS back within 5% of the T peak above noise.
        level = noise + 0.05 * (rms[tp] - noise)
        late = tp
        while late < t1 and rms[late] > level:
            late += 1
        d = np.diff(rms[tp:t1 + 1])
        te = late
        if d.size and d.min() < 0:
            ks = tp + int(np.argmin(d))
            tan = int(round(ks + (noise - rms[ks]) / float(d.min())))
            # A shallow descent extrapolates the tangent far out (e.g. to the
            # search limit at fast rates): never past the 5% crossing.
            te = min(max(tan, tp), late)
        out["t_end"] = te
    p0, p1 = on - int(0.30 * fs), on - int(0.04 * fs)
    if rr_s:
        p0 = max(p0, a - int(0.45 * rr_s * fs))
    p0 = max(p0, 0)
    if p1 - p0 >= 5:
        pp = p0 + int(np.argmax(rms[p0:p1]))
        level = noise + p_fraction * (rms[pp] - noise)
        po = pp
        while po > p0 and rms[po] > level:
            po -= 1
        out["p_on"] = po
    return out


def global_q_waves(signal: np.ndarray, rpeaks, fs: float,
                   lead_names: list[str]) -> dict[str, tuple[float, float, float, bool]]:
    """Per-lead initial Q wave on the median beat within the global QRS (D36).

    Returns ``{lead: (q_ms, q_depth_uv, r_uv, qs)}``: the initial deflection (the first
    sample more than 20 µV from the pre-QRS baseline) is a Q wave when negative, its
    duration runs to the return across the baseline; ``qs`` is an all-negative complex
    (R < 50 µV with a trough below -200 µV). Empty when no median beat can be formed.
    """
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    f = global_fiducials(X, a, fs, rr_s)
    on, off = int(f["qrs_on"]), int(f["qrs_off"])
    out: dict[str, tuple[float, float, float, bool]] = {}
    for i, name in enumerate(lead_names):
        x = X[i].astype(float)
        base = float(np.median(x[max(0, on - int(0.04 * fs)):on])) if on > 0 else float(x[0])
        y = x[on:off + 1] - base
        if y.size < 5:
            continue
        r_amp = float(max(y.max(), 0.0))
        q_ms = q_depth = 0.0
        first = np.flatnonzero(np.abs(y) > 20.0)
        if first.size and y[first[0]] < 0:
            k = int(first[0])
            while k < y.size and y[k] < 0:
                k += 1
            q_ms, q_depth = k / fs * 1000.0, float(-y[:k].min())
        qs = r_amp < 50.0 and float(y.min()) < -200.0
        out[name] = (round(q_ms, 1), round(q_depth, 1), round(r_amp, 1), bool(qs))
    return out


def _pr_baseline(x: np.ndarray, a: int, fs: float) -> float:
    """Mean of the flattest 20 ms in [R-300, R-30] ms of one lead's median beat."""
    lo, hi = max(0, a - int(0.30 * fs)), a - int(0.03 * fs)
    w20 = max(2, int(0.02 * fs))
    best, base = np.inf, 0.0
    for s0 in range(lo, max(lo + 1, hi - w20)):
        seg = x[s0:s0 + w20]
        e = float(np.ptp(seg))
        if e < best:
            best, base = e, float(np.mean(seg))
    return base


def global_st(signal: np.ndarray, rpeaks, fs: float,
              lead_names: list[str]) -> dict[str, tuple[float, float]]:
    """ST level per lead on the median beat at the global J point and J+60 ms (D48).

    Baseline = mean of the flattest 20 ms of that lead's PR segment ([R-300, R-30]
    ms). Against GE 12SL J-point ST on PTB-XL this reads MAE 28-37 uV; the per-beat
    delineator J point read 84-138 uV. Returns ``{lead: (j_uv, j60_uv)}``.
    """
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    off = int(global_fiducials(X, a, fs, rr_s)["qrs_off"])
    j60 = min(X.shape[1] - 1, off + int(0.06 * fs))
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(lead_names):
        base = _pr_baseline(X[i], a, fs)
        out[name] = (round(float(X[i][off] - base), 1), round(float(X[i][j60] - base), 1))
    return out


def global_right_precordial_st(signal: np.ndarray, rpeaks, fs: float, lead_names: list[str],
                               leads: tuple[str, ...] = ("V1", "V2"),
                               ) -> dict[str, tuple[float, float, float, float]]:
    """ST shape in the right precordial leads for the Brugada type-1 pattern (D89):
    ``{lead: (J, J+40, J+80, T min)}`` in µV against the PR baseline, J at the global QRS
    offset, T min the most negative value from J+60 ms to the global T end."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    fid = global_fiducials(X, a, fs, rr_s)
    off, te = int(fid["qrs_off"]), fid["t_end"]
    end = int(te) if te else off + int(0.35 * fs)
    n = X.shape[1]
    out: dict[str, tuple[float, float, float, float]] = {}
    for name in leads:
        if name not in lead_names:
            continue
        x = X[lead_names.index(name)] - _pr_baseline(X[lead_names.index(name)], a, fs)
        at = [float(x[min(n - 1, off + int(round(ms / 1000.0 * fs)))]) for ms in (0, 40, 80)]
        tw = x[off + int(0.06 * fs):max(off + int(0.06 * fs) + 1, end)]
        t_min = float(tw.min()) if tw.size else 0.0
        out[name] = (round(at[0], 1), round(at[1], 1), round(at[2], 1), round(t_min, 1))
    return out


def global_terminal_s(signal: np.ndarray, rpeaks, fs: float, lead_names: list[str],
                      leads: tuple[str, ...] = ("I", "V6")) -> dict[str, float]:
    """Deepest point of the last 40 ms of the QRS against the PR baseline (D90): the wide
    terminal S of a right bundle branch block in the lateral leads."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    off = int(global_fiducials(X, a, fs, rr_s)["qrs_off"])
    out: dict[str, float] = {}
    for name in leads:
        if name in lead_names:
            x = X[lead_names.index(name)]
            seg = x[max(0, off - int(0.04 * fs)):off + 1] - _pr_baseline(x, a, fs)
            if seg.size:
                out[name] = round(float(seg.min()), 1)
    return out


def global_t_amplitude(signal: np.ndarray, rpeaks, fs: float,
                       lead_names: list[str]) -> dict[str, float]:
    """Signed T-wave peak per lead on the median beat, from J+60 ms to the global
    T end, against the PR baseline (D49). Against GE 12SL T amplitude on PTB-XL:
    MAE 36-51 uV (the per-beat estimate: 59-123 uV)."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    fid = global_fiducials(X, a, fs, rr_s)
    off, te = int(fid["qrs_off"]), fid["t_end"]
    end = int(te) if te else off + int(0.35 * fs)
    out: dict[str, float] = {}
    for i, name in enumerate(lead_names):
        tw = X[i][off + int(0.06 * fs):end] - _pr_baseline(X[i], a, fs)
        if tw.size:
            out[name] = round(float(tw[np.argmax(np.abs(tw))]), 1)
    return out


def global_p_amplitude(signal: np.ndarray, rpeaks, fs: float,
                       lead_names: list[str]) -> dict[str, float]:
    """Signed P-wave peak per lead on the median beat, µV: from the global P onset to
    20 ms before QRS onset, against the last 10 ms of the PR segment (D116). The level
    before P onset is not used: at fast rates it lies on the previous T wave. Empty when
    no P onset is found (AF, junctional rhythm)."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    fid = global_fiducials(X, a, fs, rr_s)
    po, on = fid["p_on"], int(fid["qrs_on"])
    w10 = max(1, int(0.01 * fs))
    if po is None or on - 2 * w10 - po < w10:
        return {}
    out: dict[str, float] = {}
    for i, name in enumerate(lead_names):
        pw = X[i][po:on - 2 * w10] - float(np.mean(X[i][on - w10:on]))
        out[name] = round(float(pw[np.argmax(np.abs(pw))]), 1)
    return out


def global_initial_qrs_ratio(signal: np.ndarray, rpeaks, fs: float) -> float | None:
    """Delta-wave index (D59): mean summed slope energy over the first 40 ms after the
    global QRS onset divided by the QRS peak energy. Pre-excitation starts the QRS with
    a slow, slurred upstroke: PTB-XL development median 0.16 in WPW vs 0.41-0.44 in
    bundle-branch block and other records (AUC 0.89)."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return None
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    f = global_fiducials(X, a, fs, rr_s)
    on, off = int(f["qrs_on"]), int(f["qrs_off"])
    base = np.median(X[:, max(0, on - int(0.04 * fs)):max(on, 1)], axis=1, keepdims=True)
    energy = _smooth(np.sum(np.abs(np.diff(X - base, axis=1)), axis=0), max(1, int(0.01 * fs)))
    seg = energy[on:off + 1]
    n40 = int(0.04 * fs)
    if seg.size <= n40 or seg.max() <= 0:
        return None
    return round(float(seg[:n40].mean() / seg.max()), 3)


def global_qrs_amplitudes(signal: np.ndarray, rpeaks, fs: float,
                          lead_names: list[str]) -> dict[str, float]:
    """QRS peak-to-peak per lead on the median beat between the global QRS onset and
    offset, µV (D61: low voltage needs every limb or every precordial lead small)."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    f = global_fiducials(X, a, fs, rr_s)
    on, off = int(f["qrs_on"]), int(f["qrs_off"])
    if off - on < 3:
        return {}
    return {name: round(float(np.ptp(X[i][on:off + 1])), 1) for i, name in enumerate(lead_names)}


def global_qrs_rs(signal: np.ndarray, rpeaks, fs: float,
                  lead_names: list[str]) -> dict[str, tuple[float, float]]:
    """Tallest positive and deepest negative QRS deflection per lead on the median beat,
    between the global QRS onset and offset, against the PR baseline, µV:
    ``{lead: (r_uv >= 0, s_uv <= 0)}``. Peaked T is judged against the larger of the two
    (D65); Sokolow-Lyon is |S V1| + max(R V5, R V6) (D66; per beat: bias -148 µV
    against GE 12SL, median beat -96 µV)."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return {}
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    f = global_fiducials(X, a, fs, rr_s)
    on, off = int(f["qrs_on"]), int(f["qrs_off"])
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(lead_names):
        seg = X[i][on:off + 1] - _pr_baseline(X[i], a, fs)
        if seg.size:
            out[name] = (round(float(max(seg.max(), 0.0)), 1), round(float(min(seg.min(), 0.0)), 1))
    return out


def global_intervals(signal: np.ndarray, rpeaks, fs: float) -> tuple[float | None, float | None]:
    """(PR ms, QT ms) from the global fiducials, uncorrected; None where not found."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return None, None
    rp = np.asarray(rpeaks, dtype=float)
    rr_s = float(np.median(np.diff(rp))) / fs if rp.size > 2 else None
    f = global_fiducials(X, a, fs, rr_s)
    qt = (f["t_end"] - f["qrs_on"]) / fs * 1000.0 if f["t_end"] is not None else None
    pr = (f["qrs_on"] - f["p_on"]) / fs * 1000.0 if f["p_on"] is not None else None
    if qt is not None and not 200.0 <= qt <= 900.0:
        qt = None
    if pr is not None and not 60.0 <= pr <= 600.0:
        pr = None
    return pr, qt


RSR_PRIME_MIN_UV = 100.0       # D229: r' above the baseline ...
RSR_PRIME_S_FRACTION = 0.2     # ... and at least this fraction of the S depth (chosen on PTB-XL, even ecg_id)
RSR_S_MIN_UV = 100.0


def global_rsr_prime(signal: np.ndarray, rpeaks, fs: float, lead_names: list[str],
                     leads: tuple[str, ...] = ("V1", "V2")) -> str | None:
    """The first of ``leads`` whose median QRS is r-S-r' (D229): an initial r, an S below the PR baseline, then a
    second positive wave after the S within the QRS (+ 20 ms) — the incomplete RBBB pattern."""
    X, a = median_beats(signal, rpeaks, fs)
    if X is None:
        return None
    f = global_fiducials(X, a, fs, None)
    on, off = f["qrs_on"], f["qrs_off"]
    if on is None or off is None or off <= on:
        return None
    for ld in leads:
        if ld not in lead_names:
            continue
        x = X[lead_names.index(ld)]
        base = float(np.median(x[max(0, on - int(0.03 * fs)):on])) if on > 0 else float(x[0])
        q = x[on:off + int(0.02 * fs)] - base
        if q.size < 5:
            continue
        s_i = int(np.argmin(q))
        s = float(-q[s_i])
        if s_i == 0 or s_i >= q.size - 1 or s < RSR_S_MIN_UV or float(q[:s_i].max()) <= 0:
            continue
        rp = float(q[s_i:].max())
        if rp >= RSR_PRIME_MIN_UV and rp >= RSR_PRIME_S_FRACTION * s:
            return ld
    return None
