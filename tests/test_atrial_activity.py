"""Atrial activity independent of the QRS (TZ §7.1): flutter / no false alarms.

Runs the full pipeline on synthetic stimuli. Complete-AV-block coverage lives in
the same file once the dissociated P-train stimulus is validated.
"""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline


def _run(**kw):
    return run_pipeline(generate_synthetic_ecg(**kw), mode="A", provider=StubProvider())


@pytest.mark.parametrize("seed,rate,hr", [(1, 300, 100), (2, 250, 80), (3, 300, 100)])
def test_flutter_with_variable_conduction_is_detected(seed, rate, hr):
    r = _run(heart_rate_bpm=hr, flutter_atrial_rate_bpm=rate, flutter_conduction="variable",
             seed=seed)
    m = r.measurements
    assert m.atrial_rate_bpm is not None and abs(m.atrial_rate_bpm - rate) <= 0.08 * rate
    codes = r.findings.codes()
    assert "atrial_flutter" in codes
    # A non-sinus rhythm replaces the sinus_* rate codes.
    assert not codes & {"sinus_rhythm", "sinus_tachycardia", "sinus_bradycardia"}
    assert "left_atrial_enlargement" not in codes


def test_fixed_ratio_flutter_is_found_spectrally():
    # Fixed k:1 conduction locks the F-waves to R and the template subtraction removes
    # them; the spectral search on the QRS-blanked signal still sees the sawtooth.
    r = _run(heart_rate_bpm=75, flutter_atrial_rate_bpm=300, flutter_conduction="fixed", seed=3)
    m = r.measurements
    assert m.atrial_rate_bpm == pytest.approx(300, rel=0.08)
    assert any("spectrally" in n for n in m.notes)
    assert "atrial_flutter" in r.findings.codes()


@pytest.mark.parametrize("kw", [
    dict(heart_rate_bpm=75, seed=9),
    dict(heart_rate_bpm=60, seed=12),
    dict(heart_rate_bpm=40, pr_ms=160, seed=20),
    dict(heart_rate_bpm=70, rr_jitter_ms=90, seed=10),           # sinus arrhythmia
    dict(heart_rate_bpm=130, pr_ms=150, seed=32),
    dict(heart_rate_bpm=70, pr_ms=280, seed=21),                 # first-degree AV block
    dict(heart_rate_bpm=70, bbb="rbbb", seed=6),
    dict(heart_rate_bpm=70, bbb="lbbb", seed=7),
])
def test_sinus_variants_raise_no_atrial_arrhythmia(kw):
    r = _run(**kw)
    codes = r.findings.codes()
    assert not codes & {"atrial_flutter", "complete_av_block", "atrial_fibrillation"}
    assert r.measurements.av_dissociation is not True
    assert not any(a.code == "complete_av_block" for a in r.safety.alerts)


def test_atrial_fibrillation_is_not_called_flutter():
    r = _run(heart_rate_bpm=95, p_amp_scale=0.0, rr_jitter_ms=180, seed=8)
    codes = r.findings.codes()
    assert "atrial_fibrillation" in codes
    assert "atrial_flutter" not in codes
    assert not codes & {"sinus_rhythm", "sinus_tachycardia"}


@pytest.mark.parametrize("hr,atrial,seed", [(40, 90, 6), (45, 70, 5), (42, 75, 8), (38, 100, 7), (50, 85, 10)])
def test_complete_av_block_from_dissociated_p_waves(hr, atrial, seed):
    r = _run(heart_rate_bpm=hr, dissociated_atrial_rate_bpm=atrial, seed=seed)
    m = r.measurements
    assert m.heart_rate_bpm == pytest.approx(hr, abs=2)          # P waves not counted as beats
    assert m.av_dissociation is True
    assert m.atrial_rate_bpm == pytest.approx(atrial, rel=0.08)
    codes = r.findings.codes()
    assert "complete_av_block" in codes
    assert any(a.code == "complete_av_block" for a in r.safety.alerts)   # TZ §10 must-not-miss
    assert "sinus_bradycardia" not in codes and "first_degree_av_block" not in codes


def test_atrial_rate_near_twice_ventricular_is_a_documented_miss():
    # 80/38: the atrial rate sits within 10 % of 2 x ventricular, indistinguishable from
    # leftover QRS/T periodicity by the conservative harmonic guard (see atrial.py).
    r = _run(heart_rate_bpm=38, dissociated_atrial_rate_bpm=80, seed=4)
    assert r.measurements.av_dissociation is not True


def test_fundamental_peak_with_only_negative_peaks_returns_none():
    # Regression: an autocorrelation whose local maxima are all negative used to raise
    # IndexError (threshold 0.8 x top lies above top when top < 0) and crash the pipeline.
    import numpy as np

    from cardiolens.layer3_measure.atrial import fundamental_peak

    ac = -np.abs(np.sin(np.linspace(0, 40, 4000))) - 0.1
    assert fundamental_peak(ac, 75, 750) is None


def test_organised_p_veto_is_stable_at_low_beat_counts():
    from cardiolens.layer3_measure.measure import _lacks_organised_p

    # Complete block at ~40/min: one beat more at the strip edge must not flip it.
    assert not _lacks_organised_p(2 / 6, 6)
    assert not _lacks_organised_p(2 / 7, 7)
    # No organised P (AF with a slow response), or P before only 2 of 11 beats (paced).
    assert _lacks_organised_p(0.0, 7)
    assert _lacks_organised_p(2 / 11, 11)


def test_no_complete_block_alongside_atrial_fibrillation():
    # An AF finding (irregular, no organised P) and AV dissociation together were false on
    # PTB-XL (5878, 10044): a block needs organised atria; AF with a block is regular (D89).
    import numpy as np

    from cardiolens.config import load_norms
    from cardiolens.contracts.features import Intervals, Measurements
    from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
    from cardiolens.layer4_findings import detect_findings

    sig = SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500,
                               lead_names=list(STANDARD_12_LEADS), duration_s=0.02,
                               source="synthetic", patient=Patient(age_years=60, sex="male"))

    def codes(rr_cv, p_ratio):
        m = Measurements(heart_rate_bpm=60, rhythm_regular=rr_cv < 0.1, rr_cv=rr_cv,
                         p_wave_ratio=p_ratio, p_onset_ratio=0.2, pr_std_ms=80.0,
                         fwave_index=0.2, av_dissociation=True, atrial_rate_bpm=90.0,
                         intervals=Intervals(qrs_ms=90, qt_ms=380, qtc_bazett_ms=420))
        return detect_findings(sig, m, norms=load_norms("adult"), la_ra_reversed=False).codes()

    assert "complete_av_block" not in codes(rr_cv=0.4, p_ratio=0.1)      # AF found
    assert "complete_av_block" in codes(rr_cv=0.002, p_ratio=0.6)        # regular escape
