"""D204: ventricular fibrillation is read off the raw signal and raised as a critical alert —
before it, a VF strip gave no alert or "no ECG signal"."""

import numpy as np

from cardiolens.api.service import signal_from_case
from cardiolens.arrhythmia.vf import detect_vf
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline


def _vf(seed=0, f0=5.0, amp=600.0, fs=500.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(10 * fs)) / fs
    rows = []
    for k in range(12):
        f = f0 + 0.8 * np.cumsum(rng.normal(0, 0.02, t.size))
        ph = 2 * np.pi * np.cumsum(f) / fs + rng.uniform(0, 6.28)
        env = amp * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t + k))
        rows.append(env * np.sin(ph) + rng.normal(0, 15, t.size))
    return SignalRepresentation(signal=np.array(rows, np.float32), sampling_rate_hz=fs,
                                lead_names=list(STANDARD_12_LEADS), duration_s=10.0, source="synthetic",
                                patient=Patient(age_years=60, sex="male"))


def test_vf_is_a_critical_alert():
    for seed, f0 in ((0, 5.0), (1, 6.5), (3, 4.0)):
        r = run_pipeline(_vf(seed, f0), mode="A")
        assert "ventricular_fibrillation" in r.findings.codes()
        assert "ventricular_fibrillation" in [a.code for a in r.safety.alerts]


def test_fine_vf_is_raised_even_when_no_qrs_is_found():
    r = run_pipeline(_vf(2, 5.5, amp=180.0), mode="A")
    assert "ventricular_fibrillation" in [a.code for a in r.safety.alerts]


def test_organised_rhythms_are_not_vf():
    for case in ("normal", "stemi", "brady"):
        assert "ventricular_fibrillation" not in run_pipeline(signal_from_case(case), mode="A").findings.codes()
    fast = generate_synthetic_ecg(heart_rate_bpm=180, bbb="lbbb")
    assert not detect_vf(fast.signal.astype(float), fast.sampling_rate_hz)[0]


def test_a_photo_is_never_called_vf():
    sig = _vf()
    sig.provenance.digitized = True
    assert "ventricular_fibrillation" not in run_pipeline(sig, mode="A").findings.codes()
