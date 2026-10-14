"""Atrial fibrillation by fibrillatory waves (D58).

The amplitude P gate takes coarse f-waves for organised P on a third of the
PTB-XL AF records. An irregular rhythm whose QRS-blanked residual carries a high
4-9 Hz band-power index, with no conducted P, is AF by a second route.
"""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _m(rr_cv, p_ratio, fwave, p_onset=0.2, pr_std=80.0):
    return Measurements(heart_rate_bpm=90, rhythm_regular=False, rr_cv=rr_cv, p_wave_ratio=p_ratio,
                        p_onset_ratio=p_onset, pr_std_ms=pr_std, fwave_index=fwave,
                        intervals=Intervals(qrs_ms=90, qt_ms=360, qtc_bazett_ms=420))


def _codes(m):
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_fwaves_route_calls_af_when_the_p_gate_sees_organised_p():
    assert "atrial_fibrillation" in _codes(_m(0.2, 0.7, 0.6))      # P gate says organised, f-waves say AF
    assert "atrial_fibrillation" not in _codes(_m(0.2, 0.7, 0.2))  # no f-waves: the P gate stands


def test_fwaves_route_still_needs_irregularity_and_no_conducted_p():
    assert "atrial_fibrillation" not in _codes(_m(0.05, 0.7, 0.9))                       # regular
    assert "atrial_fibrillation" not in _codes(_m(0.2, 0.7, 0.9, p_onset=0.8, pr_std=10))  # conducted P


def test_fwave_index_is_measured_and_low_on_a_sinus_record():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    m = measure(sig)
    assert m.fwave_index is not None
    assert m.fwave_index < 0.45


def test_strong_f_waves_overrule_a_conducted_p_at_a_fast_rate():
    # PTB-XL 3166, AF at 150/min: the P search found an f-wave before every QRS with a
    # "stable PR" (onset 1.0, SD 29 ms) and the record read as sinus tachycardia (D81).
    def m(fwave):
        x = _m(rr_cv=0.106, p_ratio=0.68, fwave=fwave, p_onset=1.0, pr_std=29.1)
        x.heart_rate_bpm = 150.6
        return x

    codes = detect_findings(_sig(), m(1.07), norms=load_norms("adult"), la_ra_reversed=False).codes()
    assert "atrial_fibrillation" in codes and "sinus_tachycardia" not in codes
    # A tight PR (true conduction) or moderate f-waves (non-AF PTB-XL p99 is 0.59) still
    # leave the conducted P in charge.
    tight = m(1.07)
    tight.pr_std_ms = 10.0
    assert "atrial_fibrillation" not in detect_findings(
        _sig(), tight, norms=load_norms("adult"), la_ra_reversed=False).codes()
    codes = detect_findings(_sig(), m(0.6), norms=load_norms("adult"), la_ra_reversed=False).codes()
    assert "atrial_fibrillation" not in codes
    # A measured flutter rate makes those waves flutter F waves with variable conduction:
    # flutter, not AF (PTB-XL 1773, D82).
    fl = m(1.07)
    fl.atrial_rate_bpm = 297.0
    codes = detect_findings(_sig(), fl, norms=load_norms("adult"), la_ra_reversed=False).codes()
    assert "atrial_flutter" in codes and "atrial_fibrillation" not in codes
    # Children keep the conducted-P gate (pediatric sinus arrhythmia).
    child = _sig()
    child.patient = Patient(age_years=8)
    codes = detect_findings(child, m(1.07), norms=load_norms("pediatric"), la_ra_reversed=False).codes()
    assert "atrial_fibrillation" not in codes


def test_a_sinus_tachycardia_harmonic_is_not_flutter():
    # Regular sinus tachycardia: its HR harmonics read as flutter on PTB-XL (D83, D84).
    def codes(hr, atrial, mad_cv, rr_cv=None, p_ratio=0.9, fwave=0.2):
        x = _m(rr_cv=rr_cv if rr_cv is not None else mad_cv, p_ratio=p_ratio, fwave=fwave)
        x.heart_rate_bpm, x.atrial_rate_bpm, x.rr_mad_cv = hr, atrial, mad_cv
        return detect_findings(_sig(), x, norms=load_norms("adult"), la_ra_reversed=False).codes()

    assert "atrial_flutter" not in codes(105.0, 210.0, 0.01)              # 2 x HR
    assert "atrial_flutter" not in codes(98.2, 210.0, 0.027)              # 2.14 x HR (3806)
    assert "atrial_flutter" not in codes(109.0, 225.0, 0.011, rr_cv=0.09)  # ectopics inflate RR CV (12138)
    assert "atrial_flutter" not in codes(109.0, 330.0, 0.008)             # 3 x HR (17804)
    # 2 x HR above 240 (7330: 128/min, 255) is a harmonic only with sinus P waves and a
    # quiet baseline; a sawtooth (high f-wave index) at the same rates is 2:1 flutter.
    def at255(p_ratio, fwave):
        x = _m(rr_cv=0.008, p_ratio=p_ratio, fwave=fwave)
        x.heart_rate_bpm, x.atrial_rate_bpm, x.rr_mad_cv = 128.0, 255.0, 0.009
        return detect_findings(_sig(), x, norms=load_norms("adult"), la_ra_reversed=False).codes()

    assert "atrial_flutter" not in at255(0.9, 2.37)
    assert "atrial_flutter" in at255(0.9, 4.0)
    assert "atrial_flutter" in at255(0.3, 2.0)
    # 2.5 x HR on a regular rhythm is a two-beat-cycle harmonic (6818: 112.6/min, 280);
    # with alternating 2:1/3:1 conduction the RR alternates and it is flutter (D85).
    assert "atrial_flutter" not in codes(112.6, 280.4, 0.006, rr_cv=0.009, p_ratio=0.78, fwave=0.38)
    assert "atrial_flutter" in codes(110.5, 275.2, 0.04, rr_cv=0.25, p_ratio=0.67, fwave=1.33)
    # Flutter with 2:1 conduction at 300, or a harmonic-looking rate with irregular
    # (variable) conduction, still is flutter.
    assert "atrial_flutter" in codes(150.0, 300.0, 0.01, p_ratio=0.2, fwave=4.0)   # sawtooth
    assert "atrial_flutter" in codes(105.0, 210.0, 0.15)
    assert "atrial_flutter" in codes(90.0, 270.0, 0.12)
