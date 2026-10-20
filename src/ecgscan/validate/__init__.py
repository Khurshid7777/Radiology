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
    checks: dict[str, float] = field(default_factory=dict)   # identity -> residual / RMS if unrelated


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
        # Residual relative to what it would be if the leads were unrelated: normalising by
        # the target alone made digitisation noise on a small lead (aVL 0.15 mV) look like
        # a 40 % violation on a clean page.
        rms = {ld: float(np.sqrt(np.mean(cen[ld] ** 2))) for ld in leads}
        scale = max(float(np.sqrt(rms[target] ** 2 + sum((c * rms[ld]) ** 2 for ld, c in terms.items()))), 0.05)
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


def _residual(a: np.ndarray, b: np.ndarray, fs: float) -> float | None:
    """RMS(a - b) relative to RMS of the two together, after each loses its median:
    0 = the same signal, ~1 = unrelated signals, ~1.4 = opposite."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 0.5 * fs:
        return None
    a, b = a[ok] - np.median(a[ok]), b[ok] - np.median(b[ok])
    scale = max(float(np.sqrt(np.mean(a ** 2) + np.mean(b ** 2))), 0.05)
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


def rhythm_check(panels: dict[str, Trace], rhythm: Trace, fs: float) -> dict[str, float]:
    """The rhythm strip of lead II against what the panels say it must be, where they
    were recorded at the same time (SD-060): panel II itself, and (aVF - aVR) / 1.5 from
    Goldberger's identities (aVF = II - I/2, aVR = -(I + II)/2). Name -> residual."""
    if rhythm.lead != "II":
        return {}
    n = int(round(max([rhythm.start_s + len(rhythm.samples_mv) / fs]
                      + [t.start_s + len(t.samples_mv) / fs for t in panels.values()]) * fs))
    r = _on_axis(rhythm, fs, n)
    out = {}
    if "II" in panels:
        v = _residual(r, _on_axis(panels["II"], fs, n), fs)
        if v is not None:
            out["panel_II"] = round(v, 4)
    if "aVR" in panels and "aVF" in panels:
        v = _residual(r, (_on_axis(panels["aVF"], fs, n) - _on_axis(panels["aVR"], fs, n)) / 1.5, fs)
        if v is not None:
            out["aVF_aVR"] = round(v, 4)
    return out


def lead_quality(missing_fraction: float, ambiguity: float, edge_contact: float, *,
                 weights: dict | None = None) -> float:
    """0..1 from what the tracer saw: coverage, competing ink, contact with the band edge."""
    w = weights or {"ambiguity": 1.0, "edge": 0.5}
    q = (1.0 - missing_fraction) * (1.0 - w["ambiguity"] * min(1.0, ambiguity)) * (1.0 - w["edge"] * min(1.0, edge_contact))
    return float(np.clip(q, 0.0, 1.0))


@dataclass
class MeasurementCheck:
    """What the CardioLens measurement says about a digitised result (D92, D93)."""

    beats_agree: bool | None = None          # its final beats match the page columns'
    qrs_ms: float | None = None              # its QRS width
    qrs_ms_low_hump: float | None = None     # the same with the R' hump threshold lowered


def measurement_check(result, *, hump_scale: float | None = None) -> MeasurementCheck:
    """Run CardioLens on this result (once, or twice with ``hump_scale``).

    ``hump_scale``: measure again with the terminal-hump threshold of the QRS end
    (``QRS_HUMP_FRACTION``) scaled by this. A wide QRS whose R' sits near that threshold
    (right bundle branch block) reads 50 ms shorter or longer on a fraction of a pixel:
    the amplitude of a small R' is within the digitisation error, so a width that
    depends on the threshold is not a measurement to return as ``ok`` (SD-057).
    """
    from cardiolens.config import load_app_config
    from cardiolens.layer2_preprocess import preprocess
    from cardiolens.layer3_measure import measure
    from cardiolens.layer3_measure.delineation import QRS_HUMP_FRACTION
    from ecgscan.contract.cardiolens_adapter import to_signal_representation

    cfg = load_app_config()
    pp = cfg.preprocessing
    out = MeasurementCheck()
    try:
        sig = preprocess(to_signal_representation(result), mains_frequency_hz=cfg.mains_frequency_hz,
                         bandpass_low_hz=pp.bandpass_low_hz, bandpass_high_hz=pp.bandpass_high_hz,
                         target_sampling_rate_hz=pp.target_sampling_rate_hz)
        m = measure(sig)
        out.beats_agree, out.qrs_ms = m.page_beat_agreement, m.intervals.qrs_ms
        if hump_scale is not None and out.qrs_ms is not None:
            out.qrs_ms_low_hump = measure(sig, qrs_hump_fraction=QRS_HUMP_FRACTION * hump_scale).intervals.qrs_ms
    except ValueError:
        pass
    return out


def beats_agree(result) -> bool | None:
    """The CardioLens measurement on this result: do its final beats match the page
    columns' (D92)? ``None`` when there is nothing to compare (no page columns, too few
    beats) or CardioLens cannot measure it."""
    return measurement_check(result).beats_agree
