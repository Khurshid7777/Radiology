"""P-wave morphology from the median beat (D270, D271): is the P sinus-shaped?

Sinus P points down and to the left: upright in II, inverted in aVR, frontal axis about 0-75 deg. A P inverted
in II (retrograde, low atrial) or with an odd axis and an upright aVR is not sinus. Measured on the median beat
in the window 300-40 ms before the QRS onset (not earlier than 45 % of the RR), detrended against the straight
line across the window so a T-wave tail is not taken for P (as the P axis, D99). Chosen on the even Chapman
records (tools/p_morph/study.py).
"""
from __future__ import annotations

import numpy as np

_LIMB_DEG = {"I": 0.0, "II": 60.0, "III": 120.0, "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}
P_MIN_UV = 10.0          # a P peak below this is not read
P_SNR_MIN = 2.0          # peak / (5 x the window's sample-to-sample noise)
SINUS_AXIS = (-15.0, 90.0)


def p_morphology(signal: np.ndarray, lead_names: list[str], rpeaks, fs: float) -> dict | None:
    """Signed P peak and its noise ratio in II and aVR, the P axis; None when it cannot be read."""
    from cardiolens.layer3_measure.global_beat import global_fiducials, median_beats

    rp = np.asarray(rpeaks, int)
    if rp.size < 4 or "II" not in lead_names:
        return None
    X, a = median_beats(signal, rp, fs)
    if X is None:
        return None
    rr_s = float(np.median(np.diff(rp))) / fs
    on = global_fiducials(X, a, fs, rr_s)["qrs_on"]
    p0 = max(0, on - int(0.30 * fs), a - int(0.45 * rr_s * fs))
    p1 = on - int(0.04 * fs)
    if p1 - p0 < int(0.04 * fs):
        return None
    out: dict = {}
    area = {}
    for ln in set(_LIMB_DEG) & set(lead_names):
        raw = X[lead_names.index(ln)][p0:p1].astype(float)
        seg = raw - np.linspace(raw[0], raw[-1], raw.size)
        noise = float(np.std(np.diff(raw))) + 1e-6
        k = int(np.argmax(np.abs(seg)))
        out[ln] = (float(seg[k]), float(abs(seg[k]) / (noise * 5)))
        area[ln] = float(np.mean(seg))
    xs = sum(v * np.cos(np.radians(_LIMB_DEG[ln])) for ln, v in area.items())
    ys = sum(v * np.sin(np.radians(_LIMB_DEG[ln])) for ln, v in area.items())
    axis = float(np.degrees(np.arctan2(ys, xs))) if abs(xs) + abs(ys) > 1e-6 else None
    return {"II": out.get("II"), "aVR": out.get("aVR"), "axis": axis}


def non_sinus_p(pm: dict | None) -> str | None:
    """'inverted_II' / 'odd_axis' when the P is clearly not sinus-shaped, else None."""
    if not pm or pm.get("II") is None:
        return None
    p2, s2 = pm["II"]
    if p2 < -P_MIN_UV and s2 >= P_SNR_MIN:
        return "inverted_II"
    avr = pm.get("aVR")
    ax = pm.get("axis")
    # As tuned: the noise ratio is lead II's (aVR mirrors it), the aVR P only has to be upright.
    if (avr is not None and ax is not None and avr[0] > P_MIN_UV and s2 >= P_SNR_MIN
            and not SINUS_AXIS[0] <= ax <= SINUS_AXIS[1]):
        return "odd_axis"
    return None
