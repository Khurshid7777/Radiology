"""D217: ST elevation on a paced QRS raises the STEMI alert only when concordant or proportionally large."""

from cardiolens.config import load_norms
from cardiolens.contracts.features import Measurements, Morphology, STMeasurement
from cardiolens.layer4_findings.rules import paced_primary_elevation

NORMS = load_norms("adult")


def _m(*sts):
    return Measurements(heart_rate_bpm=70, morphology=Morphology(st=list(sts)))


def test_discordant_elevation_after_a_deep_s_is_expected():
    # V1-V3 of a paced (LBBB-like) QRS: 2 mm elevation after a 15 mm S wave (ratio 0.13)
    m = _m(*[STMeasurement(lead=ld, j_point_uv=200, j60_uv=220, qrs_net_uv=-1200, s_depth_uv=1500)
             for ld in ("V1", "V2", "V3")])
    assert paced_primary_elevation(m, NORMS) == []


def test_concordant_elevation_is_primary():
    m = _m(STMeasurement(lead="II", j_point_uv=150, j60_uv=150, qrs_net_uv=600, s_depth_uv=0),
           STMeasurement(lead="V2", j_point_uv=200, j60_uv=220, qrs_net_uv=-1200, s_depth_uv=1500))
    assert paced_primary_elevation(m, NORMS) == ["II"]


def test_proportional_discordant_elevation_is_primary():
    m = _m(STMeasurement(lead="V3", j_point_uv=400, j60_uv=420, qrs_net_uv=-800, s_depth_uv=1000))
    assert paced_primary_elevation(m, NORMS) == ["V3"]
