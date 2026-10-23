"""D152: 0.6-1 mm ST depression in >= 2 contiguous leads is named (st_depression_minor);
1 mm stays st_depression; non-contiguous or upsloping or pediatric is nothing."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig(age=60):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=age, sex="male"),
    )


def _codes(st: dict[str, tuple[float, float, float]], age=60):
    """``st``: lead -> (J, J+60, J+80) in uV; other leads flat."""
    rows = [STMeasurement(lead=ld, j_point_uv=v[0], j60_uv=v[1], j80_uv=v[2], qrs_net_uv=500.0)
            for ld, v in st.items()]
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=400),
                     morphology=Morphology(st=rows))
    return detect_findings(_sig(age), m, norms=load_norms("adult" if age >= 16 else "pediatric"),
                           la_ra_reversed=False).codes()


def test_minor_depression_in_two_contiguous_leads_is_named_not_diagnosed():
    c = _codes({"V5": (-60, -75, -75), "V6": (-60, -70, -72)})
    assert "st_depression_minor" in c and "st_depression" not in c
    c = _codes({"I": (-60, -70, -70), "V6": (-60, -70, -70)})          # one lateral wall
    assert "st_depression_minor" in c


def test_one_millimetre_stays_the_main_finding():
    c = _codes({"V5": (-100, -120, -125), "V6": (-100, -115, -120)})
    assert "st_depression" in c and "st_depression_minor" not in c


def test_non_contiguous_upsloping_or_a_child_is_nothing():
    assert not {"st_depression", "st_depression_minor"} & _codes({"III": (-60, -75, -75), "V2": (-60, -70, -72)})
    assert not {"st_depression", "st_depression_minor"} & _codes({"V5": (-80, -70, -10), "V6": (-80, -70, -5)})
    assert "st_depression_minor" not in _codes({"V5": (-60, -75, -75), "V6": (-60, -70, -72)}, age=10)


def test_mode_b_report_carries_the_minor_finding():
    """Point 5 of the D152 criterion: a record with only the descriptor is not 'no findings'."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.pipeline import run_pipeline

    sig = generate_synthetic_ecg(st_elevation_uv=-75.0, st_leads=["V5", "V6"], seed=5)
    res = run_pipeline(sig, mode="B")
    codes = res.findings.codes()
    assert "st_depression_minor" in codes and "st_depression" not in codes
    assert any("депрессия ST" in c.text_ru for c in res.report.summary)
