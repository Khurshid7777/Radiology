"""D252: a wide-QRS tachycardia at 100-139/min with no sign of a sinus P is 'urgent, rule out VT' (no red alert);
with a conducted P it stays tachycardia."""
import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings
from cardiolens.safety.critical import scan_critical


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"))


def _m(**kw):
    base = dict(heart_rate_bpm=118, rhythm_regular=True,
                intervals=Intervals(pr_ms=None, qrs_ms=150, qt_ms=360, qtc_bazett_ms=470))
    base.update(kw)
    return Measurements(**base)


def test_wide_without_p_is_urgent_not_sinus():
    fs = detect_findings(_sig(), _m(p_onset_ratio=0.1), norms=load_norms("adult"), la_ra_reversed=False)
    assert "wide_qrs_tachycardia" in fs.codes() and "sinus_tachycardia" not in fs.codes()
    assert "wide_qrs_tachycardia" not in [a.code for a in scan_critical(fs, _m()).alerts]


def test_conducted_p_keeps_sinus_tachycardia():
    m = _m(p_onset_ratio=0.9, pr_std_ms=8.0, intervals=Intervals(pr_ms=160, qrs_ms=150, qt_ms=360, qtc_bazett_ms=470))
    fs = detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)
    assert "sinus_tachycardia" in fs.codes() and "wide_qrs_tachycardia" not in fs.codes()


def test_narrow_qrs_stays_sinus_tachycardia():
    m = _m(p_onset_ratio=0.1, intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=320, qtc_bazett_ms=430))
    assert "sinus_tachycardia" in detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()
