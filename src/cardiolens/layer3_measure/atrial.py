"""Atrial activity independent of the QRS: flutter waves and dissociated P waves.

Fills ``Measurements.atrial_rate_bpm`` / ``av_dissociation`` (TZ §7.1), which the
atrial-flutter and complete-AV-block rules consume. Two passes:

* **Pass 1 — dissociated activity.** Average-beat subtraction removes everything
  phase-locked to the R peak (QRS, T and a *conducted* P). What survives is atrial
  activity that is *not* locked to the ventricles: flutter waves under variable
  conduction, or the independent P train of complete AV block. A normalised
  autocorrelation of the residual gives its period; a harmonic check (the peak at
  twice the lag) rejects the disorganised f-waves of atrial fibrillation.
* **Fixed k:1 flutter is not detected.** With a fixed conduction ratio the F-waves
  are locked to R and pass 1 removes them. A second pass on a QRS-only residual was
  built and evaluated on PhysioNet data: it caught 1 of 5 fixed-ratio cases while
  the P-to-T spacing of ordinary sinus rhythm mimicked 3:1 conduction, so it was
  removed. Documented limitation (TZ §16).

Every gate that fails leaves the field ``None`` and appends a note (TZ §0).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch

# Pass-1 gates.
ATRIAL_RESIDUAL_MIN_UV = 25.0      # residual RMS floor (absolute)
ATRIAL_RESIDUAL_FRACTION = 0.03    # ... and relative to the R reference
ATRIAL_AUTOCORR_MIN = 0.4          # main autocorrelation peak
ATRIAL_HARMONIC_MIN = 0.3          # peak at 2x lag (regularity)
ATRIAL_LAG_RANGE_S = (0.15, 1.5)   # 40–400 bpm
# Pass-2 gates.
# Typical flutter is 250-350/min; slow/atypical flutter (antiarrhythmics) runs
# ~200-240 and appears in the PhysioNet set, hence the wider band.
FLUTTER_BAND_BPM = (200.0, 350.0)
# Residual processing: blank +-60 ms around R (subtraction error is largest there)
# and keep the F/P-wave band only.
BLANK_MS = 60.0
ATRIAL_BAND_HZ = (2.0, 8.0)          # F-wave search (flutter)
# Dissociated-P search (complete AV block): P waves are slower and wider than
# F-waves, so a lower band and 45-200 bpm lags; a lower residual floor because P
# waves are small in V1.
P_BAND_HZ = (1.0, 6.0)
P_LAG_RANGE_S = (0.30, 1.35)
P_RESIDUAL_MIN_UV = 15.0
P_RESIDUAL_FRACTION = 0.02
# Sub-harmonic guard: the fundamental is the shortest-lag peak within this
# fraction of the highest peak.
FUNDAMENTAL_FRACTION = 0.8
# Residual periodic at k x HR (k = 1..4) is leftover QRS/T, not atrial activity.
LOCKED_HARMONICS = (1, 2, 3, 4)
# k:1 flutter with slightly variable conduction shows up in pass 1 at k x HR; it
# is accepted only with a strong, regular peak (both >= this).
FLUTTER_LOCKED_AUTOCORR_MIN = 0.5
# Spectral flutter search on the QRS-blanked signal (no template subtraction, so
# F-waves locked to the QRS survive): a sharp Welch-PSD peak in the F-wave band on
# >= 2 atrial leads agreeing within 10 %. Thresholds chosen on PTB-XL (curated
# labels): AFLT 0.63 / AFIB 0.02 / SR 0.00 at (10, 1.0); (6, 0.5) reaches 0.86 but
# flags 9 % of sinus records.
FLUTTER_SPECTRAL_BAND_HZ = (3.3, 6.0)
FLUTTER_SPECTRAL_LOW_HZ = (0.5, 3.0)
FLUTTER_SPECTRAL_SHARPNESS = 10.0     # peak / median PSD inside the band
FLUTTER_SPECTRAL_RATIO = 1.0          # band power / low-band power
# III and aVF (D82) see F waves best but also the T waves and HR harmonics of a sinus
# tachycardia: they count only with a clearer band dominance. PTB-XL (73 AFLT, 442 STACH):
# 1.5 -> 48 found / 9 false, 2.0 -> 45 / 4, 3.0 -> 42 / 3.
FLUTTER_SPECTRAL_RATIO_BY_LEAD = {"III": 2.0, "aVF": 2.0}
FLUTTER_SPECTRAL_LEADS = ("II", "III", "aVF", "V1")
FLUTTER_SPECTRAL_MIN_LEADS = 2
FWAVE_BAND_HZ = (4.0, 9.0)            # fibrillatory waves: broadband, 240-540/min
FWAVE_LOW_HZ = (0.5, 4.0)
FWAVE_LEADS = ("II", "V1")
# AV dissociation.
AV_DISSOCIATION_RATE_RANGE = (45.0, 200.0)
# Real third-degree blocks (PTB-XL) include a junctional escape at 70 with atria at
# 90 (ratio 1.29); residual periodicity within 10 % of HR is rejected separately.
AV_DISSOCIATION_MIN_RATIO = 1.2
LOCKED_TOLERANCE = 0.10            # within 10 % of k x HR = phase-locked


def template_subtract(
    x: np.ndarray, rpeaks: np.ndarray, fs: float, pre_s: float, post_s: float
) -> np.ndarray | None:
    """Subtract the median beat (aligned on R, window [-pre, +post]) from ``x``."""
    pre, post = int(pre_s * fs), int(post_s * fs)
    segs = [x[r - pre:r + post] for r in rpeaks if r - pre >= 0 and r + post <= len(x)]
    if len(segs) < 3:
        return None
    template = np.median(np.stack(segs), axis=0)
    res = x.astype(np.float64).copy()
    for i, r in enumerate(rpeaks):
        a, b = int(r) - pre, int(r) + post
        lo = a if i == 0 else max(a, (int(rpeaks[i - 1]) + int(r)) // 2)
        hi = b if i == len(rpeaks) - 1 else min(b, (int(r) + int(rpeaks[i + 1])) // 2)
        lo, hi = max(lo, 0), min(hi, len(x))
        if hi <= lo:
            continue
        ta = lo - a
        res[lo:hi] = x[lo:hi] - template[ta:ta + (hi - lo)]
    return res


def atrial_residual(
    x: np.ndarray, rpeaks: np.ndarray, fs: float, pre_s: float, post_s: float,
    band: tuple[float, float] = ATRIAL_BAND_HZ,
) -> np.ndarray | None:
    """Template-subtracted, QRS-blanked, band-limited residual (atrial band)."""
    res = template_subtract(x, rpeaks, fs, pre_s, post_s)
    if res is None:
        return None
    w = int(BLANK_MS / 1000.0 * fs)
    for r in rpeaks:
        a, b = max(0, int(r) - w), min(len(res), int(r) + w)
        if b - a > 2:
            res[a:b] = np.linspace(res[a], res[b - 1], b - a)
    nyq = fs / 2.0
    if band[1] >= nyq:
        return res
    b_, a_ = butter(2, [band[0] / nyq, band[1] / nyq], btype="band")
    return filtfilt(b_, a_, res)


def fundamental_peak(ac: np.ndarray, lmin: int, lmax: int) -> tuple[int, float] | None:
    """Shortest-lag local peak within FUNDAMENTAL_FRACTION of the highest one."""
    lmax = min(lmax, len(ac) - 2)
    if lmax <= lmin + 1:
        return None
    seg = ac[lmin:lmax + 1]
    local = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
    idx = np.flatnonzero(local) + 1
    if idx.size == 0:
        return None
    top = float(np.max(seg[idx]))
    if top <= 0.0:
        return None  # no positive periodicity at all (every local peak is negative)
    k = int(idx[np.flatnonzero(seg[idx] >= FUNDAMENTAL_FRACTION * top)[0]])
    return lmin + k, float(seg[k])


def autocorrelation(x: np.ndarray) -> np.ndarray | None:
    """Normalised autocorrelation (lag 0 == 1) via FFT."""
    y = x - float(np.mean(x))
    if float(np.std(y)) < 1e-9:
        return None
    n = len(y)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:n]
    return ac / ac[0]


def _best_local_peak(ac: np.ndarray, lmin: int, lmax: int) -> tuple[int, float] | None:
    lmax = min(lmax, len(ac) - 2)
    if lmax <= lmin + 1:
        return None
    seg = ac[lmin:lmax + 1]
    local = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
    idx = np.flatnonzero(local) + 1
    if idx.size == 0:
        return None
    k = idx[np.argmax(seg[idx])]
    return lmin + int(k), float(seg[k])


def _peak_near(ac: np.ndarray, lag: int, tol: float = 0.03) -> float:
    a, b = int(lag * (1 - tol)), int(lag * (1 + tol)) + 1
    if a >= len(ac) or a < 1:
        return 0.0
    return float(np.max(ac[a:min(b, len(ac))]))


def _blank_qrs(x: np.ndarray, rpeaks: np.ndarray, qrs_on: np.ndarray | None,
               qrs_off: np.ndarray | None, fs: float) -> np.ndarray:
    """Replace each QRS (onset-20 ms .. offset+40 ms) by a straight line."""
    y = x.astype(np.float64).copy()
    for i, r in enumerate(rpeaks):
        on_ok = qrs_on is not None and i < len(qrs_on) and np.isfinite(qrs_on[i])
        off_ok = qrs_off is not None and i < len(qrs_off) and np.isfinite(qrs_off[i])
        a = int(qrs_on[i] - 0.02 * fs) if on_ok else int(r) - int(0.06 * fs)
        b = int(qrs_off[i] + 0.04 * fs) if off_ok else int(r) + int(0.10 * fs)
        a, b = max(0, a), min(len(y), b)
        if b - a > 2:
            y[a:b] = np.linspace(y[a], y[b - 1], b - a)
    return y


def flutter_spectrum(
    leads: dict[str, np.ndarray], rpeaks: np.ndarray, fs: float,
    qrs_on: np.ndarray | None = None, qrs_off: np.ndarray | None = None,
) -> float | None:
    """Atrial rate (bpm) when >= 2 atrial leads show a sharp F-wave spectral peak."""
    hits = []
    nperseg = int(min(4.0 * fs, max(len(next(iter(leads.values()))), 2)))
    for name in FLUTTER_SPECTRAL_LEADS:
        x = leads.get(name)
        if x is None or len(x) < 2 * fs:
            continue
        y = _blank_qrs(x, rpeaks, qrs_on, qrs_off, fs)
        y -= float(np.median(y))
        f, p = welch(y, fs=fs, nperseg=nperseg, noverlap=int(0.75 * nperseg))
        band = (f >= FLUTTER_SPECTRAL_BAND_HZ[0]) & (f <= FLUTTER_SPECTRAL_BAND_HZ[1])
        low = (f >= FLUTTER_SPECTRAL_LOW_HZ[0]) & (f < FLUTTER_SPECTRAL_LOW_HZ[1])
        if not band.any() or not low.any():
            continue
        k = int(np.argmax(p[band]))
        sharp = float(p[band][k] / (np.median(p[band]) + 1e-12))
        ratio = float(p[band].sum() / (p[low].sum() + 1e-12))
        if (sharp >= FLUTTER_SPECTRAL_SHARPNESS
                and ratio >= FLUTTER_SPECTRAL_RATIO_BY_LEAD.get(name, FLUTTER_SPECTRAL_RATIO)):
            hits.append(float(f[band][k]) * 60.0)
    if len(hits) < FLUTTER_SPECTRAL_MIN_LEADS:
        return None
    med = float(np.median(hits))
    agreeing = [h for h in hits if abs(h - med) <= 0.10 * med]
    return round(float(np.median(agreeing)), 1) if len(agreeing) >= FLUTTER_SPECTRAL_MIN_LEADS else None


def fwave_index(
    leads: dict[str, np.ndarray], rpeaks: np.ndarray, fs: float,
    qrs_on: np.ndarray | None = None, qrs_off: np.ndarray | None = None,
) -> float | None:
    """Fibrillatory-wave index (D58): band power 4-9 Hz over 0.5-4 Hz of the
    QRS-blanked residual, median over II and V1. Fibrillatory waves are
    broadband in that band and leave no organised P; a sinus baseline, T waves
    and respiration sit below 4 Hz. On PTB-XL AF records the median is 0.36
    (p10-p90 0.11-1.10), on records without AF or flutter 0.25 (0.12-0.44)."""
    vals = []
    for name in FWAVE_LEADS:
        x = leads.get(name)
        if x is None or len(x) < 2 * fs:
            continue
        y = _blank_qrs(x, rpeaks, qrs_on, qrs_off, fs)
        y = y - float(np.median(y))
        nperseg = int(min(4.0 * fs, len(y)))
        f, p = welch(y, fs=fs, nperseg=nperseg, noverlap=int(0.75 * nperseg))
        band = (f >= FWAVE_BAND_HZ[0]) & (f <= FWAVE_BAND_HZ[1])
        low = (f >= FWAVE_LOW_HZ[0]) & (f < FWAVE_LOW_HZ[1])
        if band.any() and low.any():
            vals.append(float(p[band].sum() / (p[low].sum() + 1e-12)))
    return round(float(np.median(vals)), 3) if vals else None


def _locked_harmonic(rate: float, hr: float) -> int | None:
    """k when ``rate`` sits within LOCKED_TOLERANCE of k x HR, else None."""
    for k in LOCKED_HARMONICS:
        if abs(rate - k * hr) / (k * hr) <= LOCKED_TOLERANCE:
            return k
    return None


def atrial_activity(
    leads: dict[str, np.ndarray],
    rpeaks: np.ndarray,
    fs: float,
    *,
    heart_rate_bpm: float | None,
    rr_cv: float | None,
    r_ref: float,
    pr_unstable: bool = False,
    qrs_on: np.ndarray | None = None,
    qrs_off: np.ndarray | None = None,
    spectral_leads: dict[str, np.ndarray] | None = None,
) -> tuple[float | None, bool | None, list[str]]:
    """Return (atrial_rate_bpm, av_dissociation, notes).

    ``leads`` maps lead name -> signal for the atrial leads to examine (II, V1);
    the lead with the strongest periodic residual wins. ``pr_unstable`` says the
    beat-to-beat PR varies widely although a P was found before most beats — the
    signature of dissociation, which lets a P-band candidate locked at 2-4 x HR
    through (a fixed 2:1 block has a constant PR).
    """
    notes: list[str] = []
    if len(rpeaks) < 4 or heart_rate_bpm is None or not leads:
        return None, None, ["atrial activity: too few beats"]
    lmin, lmax = int(ATRIAL_LAG_RANGE_S[0] * fs), int(ATRIAL_LAG_RANGE_S[1] * fs)
    gate = max(ATRIAL_RESIDUAL_MIN_UV, ATRIAL_RESIDUAL_FRACTION * r_ref)
    rate: float | None = None
    dissociated = False

    # --- Pass 1: activity not phase-locked to the QRS (best lead wins) ---
    best = None  # (peak, harmonic, lag)
    any_signal = False
    for x in leads.values():
        res = atrial_residual(x, rpeaks, fs, 0.25, 0.45)
        if res is None or float(np.sqrt(np.mean(res * res))) < gate:
            continue
        any_signal = True
        ac = autocorrelation(res)
        peak = fundamental_peak(ac, lmin, lmax) if ac is not None else None
        if peak is None:
            continue
        lag, val = peak
        harm = _peak_near(ac, 2 * lag)
        if best is None or val > best[0]:
            best = (val, harm, lag)
    if not any_signal:
        notes.append("atrial activity: no dissociated atrial signal")
    elif best is None or best[0] < ATRIAL_AUTOCORR_MIN:
        notes.append("atrial activity: residual has no dominant period")
    elif best[1] < ATRIAL_HARMONIC_MIN:
        notes.append("atrial activity: residual not regular (no harmonic)")
    else:
        cand = 60.0 * fs / best[2]
        k = _locked_harmonic(cand, heart_rate_bpm)
        # Residual periodic at RR itself is leftover QRS/T. At k >= 2 it is either a
        # QRS/T harmonic (weak) or k:1 flutter with slightly variable conduction
        # (strong, regular, in band) — only the latter counts, and never as
        # dissociation.
        if k is None:
            rate, dissociated = round(cand, 1), True
        elif (k >= 2 and FLUTTER_BAND_BPM[0] <= cand <= FLUTTER_BAND_BPM[1]
              and best[0] >= FLUTTER_LOCKED_AUTOCORR_MIN and best[1] >= FLUTTER_LOCKED_AUTOCORR_MIN):
            rate = round(cand, 1)
        else:
            notes.append("atrial activity: residual periodic at a multiple of RR (phase-locked)")

    # --- Stage 1b: spectral F-wave search (fixed-ratio flutter locked to the QRS) ---
    if rate is None or rate < FLUTTER_BAND_BPM[0]:
        spec = flutter_spectrum(spectral_leads or leads, rpeaks, fs, qrs_on, qrs_off)
        if spec is not None and FLUTTER_BAND_BPM[0] <= spec <= FLUTTER_BAND_BPM[1]:
            rate = spec
            notes.append("atrial activity: flutter waves found spectrally")

    # --- Stage 2: dissociated P waves (complete AV block), P band / slower lags ---
    # Only when stage 1 found nothing usable. Anything periodic that survives the
    # template subtraction is not phase-locked to the ventricles by construction,
    # except leftover QRS/T errors at multiples of RR — those are rejected, which
    # also drops a true block whose atrial rate happens to sit within 10 % of
    # k x ventricular rate (documented limitation; a false "complete AV block" is
    # a critical alert, so the conservative side is chosen).
    if not dissociated:
        pmin, pmax = int(P_LAG_RANGE_S[0] * fs), int(P_LAG_RANGE_S[1] * fs)
        gate_p = max(P_RESIDUAL_MIN_UV, P_RESIDUAL_FRACTION * r_ref)
        best_p = None
        for x in leads.values():
            res = atrial_residual(x, rpeaks, fs, 0.25, 0.45, band=P_BAND_HZ)
            if res is None or float(np.sqrt(np.mean(res * res))) < gate_p:
                continue
            ac = autocorrelation(res)
            peak = fundamental_peak(ac, pmin, pmax) if ac is not None else None
            if peak is None:
                continue
            lag, val = peak
            if best_p is None or val > best_p[0]:
                best_p = (val, _peak_near(ac, 2 * lag), lag)
        if (best_p is not None and best_p[0] >= ATRIAL_AUTOCORR_MIN
                and best_p[1] >= ATRIAL_HARMONIC_MIN):
            cand = 60.0 * fs / best_p[2]
            k = _locked_harmonic(cand, heart_rate_bpm)
            if k is None or (k >= 2 and pr_unstable):
                if rate is None:
                    rate = round(cand, 1)
                dissociated = True
                if k is not None:
                    notes.append(f"atrial activity: P rate ~{k} x HR accepted (PR unstable)")
            else:
                notes.append("atrial activity: P-band residual periodic at a multiple of RR")

    # --- AV dissociation (complete AV block signal) ---
    # The escape rhythm of a real block is often irregular (PTB-XL: RR-CV 0.2-0.5 in
    # 9 of 16), so ventricular regularity is not required; the residual gates and
    # the stable-PR rejection in measure() carry the false-positive protection.
    if (dissociated and rate is not None
            and AV_DISSOCIATION_RATE_RANGE[0] <= rate <= AV_DISSOCIATION_RATE_RANGE[1]):
        av: bool | None = bool(rate >= AV_DISSOCIATION_MIN_RATIO * heart_rate_bpm)
    else:
        av = False
    return rate, av, notes
