"""D150: a PR just above the norm is named, not diagnosed; the block starts past pr_block_from."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig(age=60):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=age, sex="male"),
    )


def _codes(pr_ms, age=60):
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0, p_onset_ratio=1.0,
                     pr_std_ms=5.0, intervals=Intervals(pr_ms=pr_ms, qrs_ms=90, qt_ms=380, qtc_bazett_ms=400))
    return detect_findings(_sig(age), m, norms=load_norms("adult" if age >= 16 else "pediatric"),
                           la_ra_reversed=False).codes()


def test_pr_200_to_210_is_borderline_and_above_is_the_block():
    assert "pr_borderline" in _codes(206) and "first_degree_av_block" not in _codes(206)
    assert "first_degree_av_block" in _codes(224) and "pr_borderline" not in _codes(224)
    assert not {"pr_borderline", "first_degree_av_block"} & _codes(180)


def test_children_keep_the_single_threshold():
    assert "first_degree_av_block" in _codes(175, age=8)       # pediatric norm: PR upper 160-170 at 8 y
