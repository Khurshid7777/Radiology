"""D281: the soft VF tier — half the leads meet the VF criteria and the window was read as AF / flutter or closed by a
quality gate -> the VF / polymorphic VT alert instead of silence."""
from __future__ import annotations

import cardiolens.pipeline as pl
from cardiolens.arrhythmia.vf import vf_soft_share
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg

VF_LEAD = {"f_dom": 5.0, "conc": 0.5, "leak": 0.3, "rms": 200.0, "crest": 3.0}
QRS_LEAD = {"f_dom": 5.0, "conc": 0.1, "leak": 0.9, "rms": 200.0, "crest": 8.0}


def test_soft_share_counts_usable_leads_only():
    assert vf_soft_share([VF_LEAD, QRS_LEAD]) == 0.5
    assert vf_soft_share([VF_LEAD, None, QRS_LEAD, QRS_LEAD]) == 1 / 3
    assert vf_soft_share([None]) == 0.0 and vf_soft_share([]) == 0.0


def test_no_signal_window_with_vf_in_half_the_leads_raises_the_alert(monkeypatch):
    monkeypatch.setattr(pl, "detect_vf", lambda x, fs: (False, [VF_LEAD, QRS_LEAD]))
    monkeypatch.setattr(pl, "_hf_amplitude_uv", lambda sig: 0.0)          # -> "no ECG signal" gate
    res = pl.run_pipeline(generate_synthetic_ecg(seed=1), mode="A")
    f = res.findings.by_code("ventricular_fibrillation")
    assert f is not None and f.critical and f.label_ru.startswith("Возможна фибрилляция")
    assert "ventricular_fibrillation" in {a.code for a in res.safety.alerts}


def test_soft_tier_needs_a_gate_or_af(monkeypatch):
    # an organised sinus record with VF-like metrics in half the leads gets no alert
    monkeypatch.setattr(pl, "detect_vf", lambda x, fs: (False, [VF_LEAD, QRS_LEAD]))
    res = pl.run_pipeline(generate_synthetic_ecg(seed=2), mode="A")
    assert "ventricular_fibrillation" not in res.findings.codes()


def test_soft_tier_needs_half_the_leads(monkeypatch):
    monkeypatch.setattr(pl, "detect_vf", lambda x, fs: (False, [VF_LEAD, QRS_LEAD, QRS_LEAD]))
    monkeypatch.setattr(pl, "_hf_amplitude_uv", lambda sig: 0.0)
    res = pl.run_pipeline(generate_synthetic_ecg(seed=3), mode="A")
    assert "ventricular_fibrillation" not in res.findings.codes()
