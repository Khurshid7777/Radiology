"""D335: a wide LBBB-shaped QRS (V1, V2 negative; I or V6 positive; QRS >= 130, not from the retry search) makes discordant
ST elevation secondary (Smith-modified Sgarbossa) though no LBBB was named. RBBB, narrow QRS, the retry width, concordant
or proportional elevation keep the STEMI alert."""
import numpy as np
import pytest

from cardiolens.config import load_norms
from cardiolens.contracts.features import (
    Intervals,
    Measurements,
    Morphology,
    STMeasurement,
    Voltages,
)
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
                                duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"))


def _m(qrs_ms=150.0, j=250.0, s=1500.0, net_v1=-1500.0, retry=False, elevated=("V1", "V2", "V3"), rbbb=False):
    net = {"I": 700.0, "aVL": 500.0, "II": 300.0, "III": -200.0, "aVF": 100.0, "aVR": -500.0,
           "V1": net_v1, "V2": -1500.0, "V3": -1200.0, "V4": -300.0, "V5": 600.0, "V6": 800.0}
    st = []
    for ld in STANDARD_12_LEADS:
        e = j if ld in elevated else 0.0
        st.append(STMeasurement(lead=ld, j_point_uv=e, j60_uv=e, j80_uv=e, qrs_net_uv=net[ld],
                                s_depth_uv=s if net[ld] < 0 else 100.0, r_height_uv=100.0 if net[ld] < 0 else net[ld]))
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, rr_cv=0.03, p_wave_ratio=1.0,
                     qrs_v1_terminal_positive=True if rbbb else None, voltages=Voltages(),
                     intervals=Intervals(pr_ms=160, qrs_ms=qrs_ms, qt_ms=420, qtc_bazett_ms=440),
                     morphology=Morphology(st=st))
    m.qrs_width_from_retry = retry
    return m


def _codes(m):
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_lbbb_shaped_wide_qrs_silences_expected_discordant_elevation():
    c = _codes(_m())
    assert "lbbb" not in c                       # the block was not named: D335's case
    assert "st_elevation" not in c and "secondary_st_t_change" in c


@pytest.mark.parametrize("kw", [dict(j=450.0, s=1500.0),          # J >= 0.25 S: primary (Sgarbossa-Smith)
                                dict(qrs_ms=110.0),                # narrow
                                dict(qrs_ms=125.0),                # wide but under 130
                                dict(retry=True),                  # width from the retry search
                                dict(net_v1=900.0)])               # V1 positive (RBBB-like): not the LBBB shape
def test_stemi_alert_kept(kw):
    assert "st_elevation" in _codes(_m(**kw))


def test_rbbb_with_inferior_elevation_keeps_the_alert():
    m = _m(qrs_ms=140.0, net_v1=900.0, elevated=("II", "III", "aVF"), rbbb=True)
    assert "st_elevation" in _codes(m)
