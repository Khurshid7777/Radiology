"""Beat-by-beat P-wave search (D207).

The QRS-T complex is cancelled by a template locked to the R peak (from ``R - 60 ms`` to
``R + QT + 60 ms``); the segment before the QRS is left alone, so a conducted P stays where it is, and
a P that is not locked to the ventricles (complete AV block, a blocked P of second-degree block)
survives even where it lands on a T wave. On the residual, P waves are peaks 40-140 ms wide above the
noise of the T-P segment. Per lead the most regular P series is kept; across leads the one with the most
regular P-P wins. Each P is then paired with the next QRS:

* a conducted P precedes its QRS by a PR within ``PR_RANGE_MS``;
* a P with no QRS inside that window, on time by the P-P rhythm, is a non-conducted ("blocked") P.

The summary feeds the AV-conduction classes (AHA/ACCF/HRS 2009, part IV): complete AV block,
second-degree block (Mobitz I / II / 2:1). Nothing here is a finding by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

P_LEADS = ("II", "III", "aVF", "V1")
BAND_HZ = (1.0, 15.0)
QRS_PRE_MS = 60.0
T_POST_MS = 60.0
P_WIDTH_MS = (40.0, 140.0)
MIN_PP_S = 0.25
HEIGHT_K = 3.0                  # peak height above the residual's robust SD in the T-P segment
PR_RANGE_MS = (80.0, 400.0)     # a P this far before a QRS is conducted to it
PP_TOL = 0.15                   # |PP - n x period| / period for a regular P-P
COMBINE = False                 # attempt 2: also search the first principal component of all residuals
COMBINE_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass
class PSeries:
    lead: str
    p: list[int] = field(default_factory=list)          # P peak samples
    conducted: list[int] = field(default_factory=list)  # indices into p
    blocked: list[int] = field(default_factory=list)    # indices into p, on time and not conducted
    pr_ms: list[float] = field(default_factory=list)    # PR of each conducted P (peak to R, minus 40 ms)
    pp_period_s: float | None = None
    pp_regular: float = 0.0
    atrial_rate_bpm: float | None = None

    def summary(self, hr_bpm: float | None) -> dict:
        pr = np.asarray(self.pr_ms, dtype=float)
        iqr = float(np.subtract(*np.percentile(pr, [75, 25]))) if pr.size >= 4 else None
        return {"lead": self.lead, "n_p": len(self.p), "n_conducted": len(self.conducted),
                "n_blocked": len(self.blocked), "pp_regular": round(self.pp_regular, 3),
                "atrial_rate_bpm": None if self.atrial_rate_bpm is None else round(self.atrial_rate_bpm, 1),
                "rate_ratio": (None if self.atrial_rate_bpm is None or not hr_bpm
                               else round(self.atrial_rate_bpm / hr_bpm, 3)),
                "pr_iqr_ms": None if iqr is None else round(iqr, 1),
                "pr_median_ms": None if pr.size == 0 else round(float(np.median(pr)), 1)}


def _cancel_qrst(x: np.ndarray, rp: np.ndarray, fs: float, qt_ms: float) -> np.ndarray | None:
    a = int(QRS_PRE_MS / 1000 * fs)
    b = int((qt_ms + T_POST_MS) / 1000 * fs)
    rr = np.diff(rp)
    if rr.size and b > int(np.median(rr) * 0.95):        # never reach into the next beat's P
        b = int(np.median(rr) * 0.95)
    segs = [x[r - a:r + b] for r in rp if r - a >= 0 and r + b <= len(x)]
    if len(segs) < 3:
        return None
    tmpl = np.median(np.stack(segs), axis=0)
    y = x.astype(np.float64).copy()
    for r in rp:
        lo, hi = r - a, r + b
        if lo >= 0 and hi <= len(y):
            y[lo:hi] -= tmpl
    return y


def _series(y: np.ndarray, rp: np.ndarray, fs: float, qrs_ms: float, lead: str) -> PSeries | None:
    from scipy.signal import find_peaks, peak_widths

    blank = np.zeros(len(y), dtype=bool)
    half = int(max(0.06, qrs_ms / 2000.0 + 0.03) * fs)
    for r in rp:
        blank[max(0, r - half):r + half] = True
    quiet = y[~blank]
    if quiet.size < fs:
        return None
    sd = 1.4826 * float(np.median(np.abs(quiet - np.median(quiet)))) + 1e-9
    best = None
    for pol in (1.0, -1.0):
        z = pol * y
        z[blank] = 0.0
        pk, _ = find_peaks(z, height=HEIGHT_K * sd, prominence=HEIGHT_K * sd, distance=int(MIN_PP_S * fs))
        if pk.size < 3:
            continue
        w = peak_widths(z, pk, rel_height=0.5)[0] / fs * 1000.0
        keep = (w >= P_WIDTH_MS[0] / 2) & (w <= P_WIDTH_MS[1])
        pk = pk[keep]
        if pk.size < 3:
            continue
        pp = np.diff(pk) / fs
        period = float(np.median(pp[pp <= 1.6 * np.percentile(pp, 25)]))
        n = np.round(pp / period)
        regular = float(np.mean((n == 1) & (np.abs(pp - period) / period <= PP_TOL)))
        if best is None or (regular, pk.size) > (best.pp_regular, len(best.p)):
            best = PSeries(lead=lead, p=[int(v) for v in pk], pp_period_s=period, pp_regular=regular,
                           atrial_rate_bpm=60.0 / period)
    if best is None:
        return None
    lo, hi = PR_RANGE_MS[0] / 1000 * fs, PR_RANGE_MS[1] / 1000 * fs
    p = np.asarray(best.p)
    period = best.pp_period_s or 1.0
    for k, pi in enumerate(p):
        nxt = rp[rp > pi]
        if nxt.size == 0:
            continue
        d = nxt[0] - pi
        # the nearest P before that QRS is the conducted one
        later = p[(p > pi) & (p < nxt[0])]
        if lo <= d <= hi and later.size == 0:
            best.conducted.append(k)
            best.pr_ms.append(d / fs * 1000.0 - 40.0)
        elif k > 0 and abs((pi - p[k - 1]) / fs - period) <= PP_TOL * period:
            best.blocked.append(k)
    return best


def p_wave_series(sig, rpeaks, qrs_ms: float | None, qt_ms: float | None) -> PSeries | None:
    """The most regular beat-by-beat P series over the P leads, or None."""
    from scipy.signal import butter, filtfilt

    fs = float(sig.sampling_rate_hz)
    rp = np.asarray(rpeaks or [], dtype=int)
    if rp.size < 4:
        return None
    nyq = fs / 2.0
    b, a = butter(2, [BAND_HZ[0] / nyq, min(BAND_HZ[1] / nyq, 0.99)], btype="band")
    out = None
    resid = []
    for lead in COMBINE_LEADS if COMBINE else P_LEADS:
        if lead not in sig.lead_names:
            continue
        x = filtfilt(b, a, np.asarray(sig.lead(lead), dtype=np.float64))
        y = _cancel_qrst(x, rp, fs, qt_ms or 400.0)
        if y is None:
            continue
        resid.append(y)
        if lead not in P_LEADS:
            continue
        s = _series(y, rp, fs, qrs_ms or 100.0, lead)
        if s is not None and (out is None or (s.pp_regular, len(s.p)) > (out.pp_regular, len(out.p))):
            out = s
    if COMBINE and len(resid) >= 3:
        # First principal component of the residuals (D207, attempt 2): a small P seen in several
        # leads at once rises above each lead's own noise.
        r = np.vstack([v for v in resid if np.isfinite(v).all()]) if resid else np.empty((0, 0))
        if r.shape[0] < 3:
            return out
        r = r - r.mean(axis=1, keepdims=True)
        r = r / (np.std(r, axis=1, keepdims=True) + 1e-9)
        _, v = np.linalg.eigh(r @ r.T / r.shape[1])          # leads x leads: small and stable
        pc = v[:, -1] @ r
        s = _series(pc, rp, fs, qrs_ms or 100.0, "PC1")
        if s is not None and (out is None or (s.pp_regular, len(s.p)) > (out.pp_regular, len(out.p))):
            out = s
    return out
