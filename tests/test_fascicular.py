"""D202: fascicular blocks, bifascicular blocks and incomplete RBBB from the limb-lead QRS shape
and the axis (AHA/ACCF/HRS 2009, Surawicz et al., part III)."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Axes, Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings

LAFB, LPFB, IRBBB = "left_anterior_fascicular_block", "left_posterior_fascicular_block", "incomplete_rbbb"


def _sig(age_years=55):
    return SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500,
                                lead_names=list(STANDARD_12_LEADS), duration_s=0.02, source="synthetic",
                                patient=Patient(age_years=age_years, sex="male"))


def _m(axis, qrs=96, rs=None, q=(), v1_terminal=None, v1=(300, 900)):
    """rs: lead -> (R, S) µV; leads not given are a plain small R. q: leads with a Q >= 30 ms."""
    shape = {ld: (600.0, 200.0) for ld in STANDARD_12_LEADS}
    shape["V1"] = v1
    shape.update(rs or {})
    st = [STMeasurement(lead=ld, r_height_uv=r, s_depth_uv=s) for ld, (r, s) in shape.items()]
    qw = {ld: ((30.0, 150.0, shape[ld][0], False) if ld in q else (0.0, 0.0, shape[ld][0], False))
          for ld in STANDARD_12_LEADS}
    return Measurements(heart_rate_bpm=72, rhythm_regular=True, qrs_v1_terminal_positive=v1_terminal,
                        intervals=Intervals(pr_ms=160, qrs_ms=qrs, qt_ms=390, qtc_bazett_ms=410),
                        axes=Axes(qrs_deg=axis), morphology=Morphology(st=st, q_waves=qw))


def _codes(m, age=55):
    return _find(m, age).codes()


def _find(m, age=55):
    return detect_findings(_sig(age), m, norms=load_norms("adult"), la_ra_reversed=False)


LAFB_SHAPE = {"aVL": (1200, 100), "II": (300, 900), "III": (200, 1200), "aVF": (250, 1000), "I": (900, 200)}
LPFB_SHAPE = {"I": (200, 700), "aVL": (150, 600), "III": (1200, 100), "aVF": (1000, 150), "II": (900, 200)}


def test_left_anterior_fascicular_block():
    assert LAFB in _codes(_m(-60, rs=LAFB_SHAPE, q={"aVL"}))


def test_a_normal_axis_is_not_a_fascicular_block():
    codes = _codes(_m(40))
    assert not codes & {LAFB, LPFB, IRBBB}


def test_lafb_with_a_complete_rbbb_is_named_bifascicular():
    f = next(x for x in _find(_m(-60, qrs=140, rs=LAFB_SHAPE, q={"aVL"}, v1_terminal=True)).findings
             if x.code == LAFB)
    assert "двухпучковая" in f.label_ru


def test_left_posterior_fascicular_block_in_an_adult_only():
    m = _m(120, rs=LPFB_SHAPE, q={"III", "aVF"})
    assert LPFB in _codes(m)
    assert LPFB not in _codes(m, age=10)          # a right axis is normal in children


def test_lpfb_is_not_called_over_right_ventricular_hypertrophy():
    assert LPFB not in _codes(_m(120, rs=LPFB_SHAPE, q={"III", "aVF"}, v1=(1100, 300)))


def test_no_fascicular_block_is_named_inside_an_lbbb():
    codes = _codes(_m(-60, qrs=160, rs=LAFB_SHAPE, q={"aVL"}, v1_terminal=False))
    assert "lbbb" in codes and LAFB not in codes


def test_an_axis_between_minus_30_and_minus_45_is_a_possible_lafb():
    f = next(x for x in _find(_m(-38, rs=LAFB_SHAPE)).findings if x.code == LAFB)
    assert f.label_ru.startswith("Возможна") and f.confidence < 0.7


def test_an_old_inferior_infarct_is_not_an_lafb():
    assert LAFB not in _codes(_m(-60, rs=LAFB_SHAPE, q={"aVL", "III", "aVF"}))


def test_a_nonspecific_wide_qrs_keeps_the_fascicular_axis():
    codes = _codes(_m(-60, qrs=128, rs=LAFB_SHAPE, v1_terminal=None))
    assert "wide_qrs_nonspecific" in codes and LAFB in codes


def test_incomplete_rbbb_is_an_rsr_in_v1_v2_below_the_wide_qrs():
    """D229: the r' after the S on the median beat (``rsr_prime_lead``), QRS 80-110 ms."""
    def m(qrs, lead):
        x = _m(40, qrs=qrs, v1_terminal=lead is not None)
        x.rsr_prime_lead = lead
        return x
    assert IRBBB in _codes(m(100, "V1")) and IRBBB in _codes(m(85, "V2"))
    assert IRBBB not in _codes(m(100, None))
    codes = _codes(m(140, "V1"))
    assert "rbbb" in codes and IRBBB not in codes


def test_a_slow_wide_rhythm_asks_to_exclude_complete_block():
    m = _m(-10, qrs=150, v1_terminal=False)
    m.heart_rate_bpm = 38
    assert "slow_wide_escape_rhythm" in _codes(m)
    m.heart_rate_bpm = 55
    assert "slow_wide_escape_rhythm" not in _codes(m)


def test_a_slow_rhythm_without_a_conducted_p_asks_too():
    m = _m(40, qrs=90)
    m.heart_rate_bpm, m.pr_std_ms = 38, None
    assert "slow_wide_escape_rhythm" in _codes(m)
    m.pr_std_ms = 12.0
    assert "slow_wide_escape_rhythm" not in _codes(m)


def test_rsr_prime_detector_on_a_drawn_median_beat():
    """D229: r-S-r' in V1 is found; a plain rS is not."""
    import numpy as np

    from cardiolens.layer3_measure.global_beat import global_rsr_prime

    fs, n = 500.0, 5000
    t = np.arange(n) / fs
    rp = list(range(250, n - 400, 400))
    sig = np.zeros((12, n))
    def g(c, w, a):
        return a * np.exp(-((t[:, None] - c) / w) ** 2).sum(axis=1)
    for li in range(12):
        sig[li] = g(np.array(rp) / fs, 0.010, 900)          # an R in every lead
    v1 = 6
    centers = np.array(rp) / fs
    sig[v1] = g(centers - 0.020, 0.008, 250) + g(centers + 0.005, 0.010, -800) + g(centers + 0.040, 0.010, 400)
    names = list(STANDARD_12_LEADS)
    assert global_rsr_prime(sig, rp, fs, names) == "V1"
    sig[v1] = g(centers - 0.020, 0.008, 250) + g(centers + 0.005, 0.012, -800)
    sig[names.index("V2")] = sig[v1]
    assert global_rsr_prime(sig, rp, fs, names) is None
