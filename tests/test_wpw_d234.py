"""D234: a definite-LBBB width is pre-excitation when the delta wave reaches V1-V3 (PR up to 180 ms)."""
import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings
from cardiolens.pipeline import _PACED_EXCLUDES


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=40, sex="male"))


def _codes(v13: float, pr: float = 130) -> set[str]:
    by = {ln: 60.0 for ln in STANDARD_12_LEADS} | {"V1": v13, "V2": v13, "V3": v13}
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=pr, qrs_ms=145, qt_ms=420, qtc_bazett_ms=450),
                     morphology=Morphology(delta_ms=60.0, delta_ms_by_lead=by))
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_delta_in_v1_v3_is_preexcitation_not_lbbb():
    codes = _codes(66.0)
    assert "wpw_preexcitation" in codes and "lbbb" not in codes


def test_delta_in_v1_v3_allows_a_longer_pr():
    assert "wpw_preexcitation" in _codes(66.0, pr=170)
    assert "wpw_preexcitation" not in _codes(66.0, pr=200)


def test_fast_start_in_v1_v3_stays_lbbb():
    codes = _codes(22.0)
    assert "lbbb" in codes and "wpw_preexcitation" not in codes


def test_paced_rhythm_withholds_wpw():
    assert "wpw_preexcitation" in _PACED_EXCLUDES
