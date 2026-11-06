"""Ventricular fibrillation / flutter from the raw signal (D204).

VF has no QRS complexes, so a detector that first looks for R peaks misses it — on a VF strip the
pipeline used to say "no ECG signal" or nothing at all. VF is a continuous, nearly sinusoidal
oscillation at 3-9 Hz (ventricular flutter: a regular sine wave at ~4-5 Hz, VF: coarse to fine
irregular waves), with no isoelectric baseline. Per lead:

* dominant frequency of the 1-30 Hz band in 2.5-9.5 Hz;
* spectral concentration: share of the 1-30 Hz power within +-1 Hz of that peak and its harmonic;
* leakage (Kuo & Dillman 1978; Amann et al. 2005): a sinusoid of period T added to itself shifted
  by T/2 cancels — sum|x[n] + x[n - T/2]| / sum(|x[n]| + |x[n - T/2]|) is small for VF/VFL and
  close to 1 for a QRS train;
* amplitude: RMS of at least ``VF_RMS_MIN_UV`` (fine VF is >= ~100 µV peak to peak);
* crest factor: no spikes above the waves — atrial flutter with conducted QRS has F waves at
  4-5 Hz too, but its QRS stands well above them.

The record is VF when at least 70 % of its leads (and at least one) satisfy all five. Thresholds:
DECISIONS D204, chosen on MIT-BIH VFDB + CUDB 10-s windows and the 3 662 PTB-XL bench records.
"""

from __future__ import annotations

import numpy as np

VF_BAND_HZ = (2.5, 9.5)
VF_LEAKAGE_MAX = 0.5
VF_CONCENTRATION_MIN = 0.3
VF_RMS_MIN_UV = 60.0
VF_LEAD_SHARE_MIN = 0.7
VF_CREST_MAX = 4.5         # 99.5th percentile |x| / RMS: QRS spikes above flutter waves raise it


def lead_metrics(x: np.ndarray, fs: float) -> dict | None:
    """Dominant frequency, spectral concentration, leakage and RMS of one lead (µV)."""
    from scipy.signal import butter, filtfilt, welch

    x = np.asarray(x, dtype=np.float64)
    if x.size < int(4 * fs) or not np.isfinite(x).all():
        return None
    nyq = fs / 2.0
    b, a = butter(2, [1.0 / nyq, min(30.0, 0.45 * fs) / nyq], btype="band")
    y = filtfilt(b, a, x)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms == 0:
        return None
    f, p = welch(y, fs=fs, nperseg=min(y.size, int(4 * fs)))
    band = (f >= 1.0) & (f <= 30.0)
    total = float(p[band].sum()) or 1e-12
    inband = (f >= VF_BAND_HZ[0]) & (f <= VF_BAND_HZ[1])
    if not inband.any():
        return None
    f_dom = float(f[inband][np.argmax(p[inband])])
    near = (np.abs(f - f_dom) <= 1.0) | (np.abs(f - 2 * f_dom) <= 1.0)
    conc = float(p[near & band].sum() / total)
    half = max(1, int(round(fs / f_dom / 2.0)))
    num = np.abs(y[half:] + y[:-half]).sum()
    den = (np.abs(y[half:]) + np.abs(y[:-half])).sum() or 1e-12
    crest = float(np.percentile(np.abs(y), 99.5) / rms)
    return {"f_dom": f_dom, "conc": conc, "leak": float(num / den), "rms": rms, "crest": crest}


def lead_is_vf(mt: dict | None) -> bool:
    return (mt is not None and mt["leak"] <= VF_LEAKAGE_MAX and mt["conc"] >= VF_CONCENTRATION_MIN
            and mt["rms"] >= VF_RMS_MIN_UV and mt["crest"] <= VF_CREST_MAX
            and VF_BAND_HZ[0] <= mt["f_dom"])


def detect_vf(signal: np.ndarray, fs: float) -> tuple[bool, list[dict | None]]:
    """(VF?, per-lead metrics) for a leads x samples array in µV."""
    mts = [lead_metrics(row, fs) for row in np.atleast_2d(signal)]
    usable = [m for m in mts if m is not None]
    if not usable:
        return False, mts
    hits = sum(lead_is_vf(m) for m in usable)
    return bool(hits >= 1 and hits / len(usable) >= VF_LEAD_SHARE_MIN), mts
