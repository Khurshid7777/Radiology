"""Layer 2 — signal quality and electrode checks (TZ §5).

Per-lead SQI, lead-off detection, and LA/RA electrode-reversal detection. The
reversal detector emits a *hypothesis* (finding code ``lead_reversal_la_ra``)
that Layer 5 treats as an artifact explanation (TZ §5, §8.3) rather than
fabricating pathology.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import Quality, SignalRepresentation

# A lead whose whole span is under this is not a recording: a lead-off electrode
# or an unrecorded channel stored as zeros (ecg-arrhythmia has limb-only
# paediatric records written as 12 leads with V1-V6 all zero — 673 of 971 records
# aged 4-8 y). Layer 3 measures nothing in such a lead (D45).
LEAD_OFF_PTP_UV = 10.0


def is_lead_off(row: np.ndarray) -> bool:
    return float(np.ptp(np.asarray(row, dtype=np.float64))) < LEAD_OFF_PTP_UV


def lead_off_leads(sig: SignalRepresentation) -> set[str]:
    """Names of the leads that carry no signal (see ``LEAD_OFF_PTP_UV``)."""
    return {name for name, row in zip(sig.lead_names, sig.signal) if is_lead_off(row)}


def per_lead_sqi(sig: SignalRepresentation) -> list[float]:
    """Heuristic per-lead SQI in [0, 1] (TZ §5).

    Combines: flatline penalty (lead-off), and a crude SNR proxy from the ratio
    of in-band structured energy to high-frequency jitter.
    """
    scores: list[float] = []
    for i in range(sig.n_leads):
        row = sig.signal[i].astype(np.float64)
        if is_lead_off(row):  # essentially flat (lead-off / unrecorded channel)
            scores.append(0.0)
            continue
        # High-frequency noise proxy: std of sample-to-sample difference.
        hf = float(np.std(np.diff(row)))
        sig_std = float(np.std(row)) + 1e-9
        snr = sig_std / (hf + 1e-9)
        score = float(np.clip(snr / (snr + 1.0), 0.0, 1.0))
        scores.append(round(score, 3))
    return scores


def detect_lead_off(sqi: list[float], threshold: float = 0.05) -> list[str]:
    """Flag near-flat leads as lead-off (TZ §5)."""
    return [f"lead_off:{i}" for i, s in enumerate(sqi) if s <= threshold]


def detect_la_ra_reversal(sig: SignalRepresentation) -> bool:
    """Detect LA/RA electrode reversal (TZ §5).

    Classic signature: lead I becomes globally inverted and aVR looks like a
    normal lead. We test for a predominantly negative P/QRS in lead I combined
    with an upright aVR — a predictable, non-physiological pattern (TZ §8.3).
    Requires the standard limb leads to be present.
    """
    names = sig.lead_names
    if not {"I", "aVR"}.issubset(set(names)):
        return False
    lead_i = sig.lead("I").astype(np.float64)
    avr = sig.lead("aVR").astype(np.float64)
    # Dominant (largest-magnitude) deflection sign, robust to baseline.
    i_dominant = lead_i[np.argmax(np.abs(lead_i - np.median(lead_i)))] - np.median(lead_i)
    avr_dominant = avr[np.argmax(np.abs(avr - np.median(avr)))] - np.median(avr)
    # Normal: lead I dominant deflection positive, aVR negative.
    # Reversal: lead I dominant negative AND aVR dominant positive.
    return bool(i_dominant < 0 and avr_dominant > 0)


def assess_quality(
    sig: SignalRepresentation, *, sqi_accept_threshold: float
) -> tuple[Quality, bool]:
    """Populate ``sig.quality`` and return (quality, la_ra_reversed)."""
    sqi = per_lead_sqi(sig)
    flags: list[str] = []
    flags.extend(detect_lead_off(sqi))
    low = [i for i, s in enumerate(sqi) if s < sqi_accept_threshold]
    if low:
        flags.append(f"low_sqi_leads:{','.join(map(str, low))}")

    swap = detect_limb_swap(sig)
    reversed_la_ra = swap == "la_ra"
    if reversed_la_ra:
        flags.append("suspected_la_ra_reversal")
    elif swap == "ra_ll":
        flags.append("suspected_ra_ll_reversal")

    quality = Quality(per_lead_sqi=sqi, flags=flags)
    sig.quality = quality
    return quality, reversed_la_ra


# D214: I and II predicted from V1-V6 (least squares on the even half of PTB-XL 1 714, 0.5-40 Hz, 100 Hz).
# The chest leads do not change when two limb electrodes are exchanged, so the swap that brings the
# recorded limb leads back into line with this prediction is the one that happened.
_SWAP_COEF_I = (-0.4839, 0.1750, 0.0209, -0.0898, 0.2254, 0.0986)
_SWAP_COEF_II = (0.0350, -0.1253, 0.0749, 0.1306, 0.0992, 0.2553)
_SWAP_MARGIN = 1.15                      # sum of the two correlations; chosen on the even half (D214)
_SWAP_NAMED = ("la_ra", "ra_ll")         # la_ll competes but is not named (low specificity, books D212)


def swap_limb_leads(lead_i: np.ndarray, lead_ii: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(I, II) as recorded with ``kind`` electrodes exchanged; an involution, so it also undoes the swap."""
    e = {"RA": np.zeros_like(lead_i), "LA": lead_i, "LL": lead_ii}
    if kind != "none":
        a, b = {"la_ra": ("LA", "RA"), "ra_ll": ("RA", "LL"), "la_ll": ("LA", "LL")}[kind]
        e[a], e[b] = e[b], e[a]
    return e["LA"] - e["RA"], e["LL"] - e["RA"]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def limb_swap_scores(sig: SignalRepresentation) -> dict[str, float] | None:
    """Fit of each swap hypothesis (higher = better), or None without the needed leads."""
    from scipy.signal import butter, filtfilt

    need = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    if not set(need) <= set(sig.lead_names):
        return None
    fs = float(sig.sampling_rate_hz)
    rows = np.array([sig.lead(ld).astype(np.float64) for ld in need])
    if rows.shape[1] < int(3 * fs) or fs < 100:
        return None
    if any(is_lead_off(r) for r in rows):
        return None
    b, a = butter(2, [0.5 / (fs / 2), 40.0 / (fs / 2)], "band")
    rows = filtfilt(b, a, rows, axis=1)[:, :: max(1, round(fs / 100.0))]
    v = rows[2:]
    pred_i, pred_ii = np.asarray(_SWAP_COEF_I) @ v, np.asarray(_SWAP_COEF_II) @ v
    out = {}
    for kind in ("none", "la_ra", "ra_ll", "la_ll"):
        ti, tii = swap_limb_leads(rows[0], rows[1], kind)
        out[kind] = _corr(ti, pred_i) + _corr(tii, pred_ii)
    return out


def detect_limb_swap(sig: SignalRepresentation) -> str | None:
    """'la_ra', 'ra_ll' or None (D214). Without the chest leads, the TZ §5 LA/RA sign is used."""
    sc = limb_swap_scores(sig)
    if sc is None:
        return "la_ra" if detect_la_ra_reversal(sig) else None
    best = max(("la_ra", "ra_ll", "la_ll"), key=lambda k: sc[k])
    return best if best in _SWAP_NAMED and sc[best] - sc["none"] > _SWAP_MARGIN else None
