"""What the page draws beside the numbers (D210, UI phase 3): the norms and the waveform samples.

Nothing here computes anything new. The norms are the same ranges the rules use for this patient,
read from the same tables (``configs/norms``). The samples are those of the waveform SVG: the same
leads, the same window, thinned to 250 Hz. A photo under a reliability gate gets no samples, so a
ruler is never laid over a trace that is not to be trusted.
"""

from __future__ import annotations

from typing import Any

from cardiolens.config import norms_for_patient

GATED = ("photo_unreliable", "no_ecg_signal", "amplitude_implausible")
SAMPLES_HZ = 250.0


def display_norms(result) -> dict[str, Any] | None:
    """{field: [low, high]} for HR, PR, QRS, QTc and the QRS axis; a missing bound is None."""
    from cardiolens.layer4_findings.rules import _axis_range, _hr_range, _interval_norms

    p = result.signal.patient
    try:
        norms = norms_for_patient(p.age_years, p.age_days)
    except Exception:
        return None
    iv = _interval_norms(norms, p)
    hr = _hr_range(norms, p)
    axis = _axis_range(norms, p)
    qtc = norms.get("qtc_ms", {})
    upper = {"male": qtc.get("male_upper"), "female": qtc.get("female_upper")}.get(p.sex or "")
    out = {
        "heart_rate_bpm": list(hr) if hr else None,
        "pr_ms": list(iv["pr"]) if iv.get("pr") else None,
        "qrs_ms": list(iv["qrs"]) if iv.get("qrs") else None,
        "qtc_ms": [qtc.get("short_lower"), upper] if upper else None,
        "qrs_axis_deg": list(axis) if axis else None,
        "table": norms.get("age_group"),
    }
    return out


def waveform_samples(result) -> dict[str, Any]:
    """The waveform SVG's leads as numbers: {lead: {"t0": s, "uv": [...]}} at SAMPLES_HZ."""
    from cardiolens.outputs.waveform import finding_leads

    codes = result.findings.codes()
    if codes & set(GATED):
        return {"gated": True, "hz": SAMPLES_HZ, "leads": {}, "marked": {}}
    sig = result.signal
    fs = sig.sampling_rate_hz
    leads, marked = finding_leads(result)
    leads = leads or [ln for ln in ("I", "II", "V1", "V5") if ln in sig.lead_names]
    total = sig.n_samples / fs
    windows = sig.provenance.lead_windows or {}
    step = max(1, round(fs / SAMPLES_HZ))
    out = {}
    for lead in leads:
        t0, t1 = windows.get(lead, (0.0, total))
        t1 = min(t1, t0 + 5.0, total)
        i0, i1 = int(t0 * fs), max(int(t0 * fs) + 1, int(t1 * fs))
        row = sig.lead(lead)[i0:i1:step]
        out[lead] = {"t0": round(i0 / fs, 4), "uv": [round(float(v)) for v in row]}
    return {"gated": False, "hz": fs / step, "leads": out, "marked": marked}
