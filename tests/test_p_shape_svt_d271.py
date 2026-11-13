"""D270 / D271: a P inverted in II (or an odd axis with an upright aVR P) is not sinus; the core's own sinus P
(found before most beats, stable PR, sinus axis) vetoes the reading."""
from cardiolens.contracts.features import Measurements
from cardiolens.layer3_measure.p_morph import non_sinus_p
from cardiolens.layer4_findings.rules import _core_sees_sinus_p


def test_non_sinus_p_kinds():
    assert non_sinus_p({"II": (-80.0, 3.0), "aVR": (40.0, 2.0), "axis": -110.0}) == "inverted_II"
    assert non_sinus_p({"II": (30.0, 2.5), "aVR": (25.0, 2.0), "axis": 130.0}) == "odd_axis"
    assert non_sinus_p({"II": (90.0, 4.0), "aVR": (-60.0, 3.0), "axis": 55.0}) is None
    assert non_sinus_p({"II": (-8.0, 3.0), "aVR": None, "axis": None}) is None        # too small
    assert non_sinus_p({"II": (-80.0, 1.0), "aVR": None, "axis": None}) is None       # in the noise
    assert non_sinus_p(None) is None


def test_core_sinus_p_vetoes():
    m = Measurements()
    m.p_onset_ratio, m.pr_std_ms = 0.9, 12.0
    m.axes.p_deg = 55.0
    assert _core_sees_sinus_p(m)
    m.axes.p_deg = -100.0
    assert not _core_sees_sinus_p(m)
    m.axes.p_deg, m.pr_std_ms = 55.0, 45.0
    assert not _core_sees_sinus_p(m)
