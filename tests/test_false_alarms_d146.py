"""D146: false alarms on normal ECGs — T inversion in V1, flat T in III/V1/aVR, long QT in
atrial fibrillation (the two bradycardia levels were rolled back on the holdout)."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer4_findings.rules import _qt_findings


def _sig(age=60):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=age, sex="male"),
    )


def _codes(hr=70, inverted=(), flat=(), age=60):
    m = Measurements(heart_rate_bpm=hr, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     intervals=Intervals(qrs_ms=90, qt_ms=380, qtc_bazett_ms=400),
                     morphology=Morphology(inverted_t_leads=list(inverted), flat_t_leads=list(flat)))
    norms = load_norms("adult" if age >= 16 else "pediatric")
    return detect_findings(_sig(age), m, norms=norms, la_ra_reversed=False).codes()


def test_t_inversion_in_an_adults_v1_is_not_counted():
    assert "t_inversion" not in _codes(inverted=["III", "V1"])        # the commonest false set
    assert "t_inversion" not in _codes(inverted=["aVR", "V1"])
    assert "t_inversion" in _codes(inverted=["V1", "V2", "V3"])       # V2-V3 still count


def test_flat_t_in_iii_v1_avr_is_normal():
    assert "flat_t_waves" not in _codes(flat=["III", "V1", "aVR", "aVL"])
    assert "flat_t_waves" in _codes(flat=["I", "aVL", "V6"])


def test_bradycardia_stays_below_60_after_the_rolled_back_change():
    """C was rolled back on PTB-XL fold 10 (D146): SBRAD is labelled at 50-59 there too."""
    assert "sinus_bradycardia" in _codes(hr=55) and "sinus_bradycardia" in _codes(hr=45)
    assert "sinus_bradycardia" not in _codes(hr=65)


def _qt_in_af(qtc_ms, af=True):
    fs = FindingSet()
    if af:
        fs.findings.append(Finding(code="atrial_fibrillation", group="rhythm", label_ru="ФП", label_uz="BF",
                                   confidence=0.9))
    iv = Intervals(qrs_ms=90, qt_ms=qtc_ms, qtc_bazett_ms=qtc_ms)
    _qt_findings(iv, load_norms("adult"), False, _sig(), fs)
    return fs


def test_long_qt_in_af_needs_500_and_says_it_is_approximate():
    fs = _qt_in_af(480)
    assert "long_qt" not in fs.codes() and "qtc_unreliable_af" in fs.codes()
    f = _qt_in_af(520).by_code("long_qt")                              # the alert is kept (TZ §10)
    assert f is not None and "ориентировочно" in f.label_ru
    assert "long_qt" in _qt_in_af(480, af=False).codes()              # without AF: unchanged
