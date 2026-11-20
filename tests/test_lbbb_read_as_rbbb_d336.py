"""D336: complete LBBB typed as RBBB because the discordant ST elevation in V1 fills the terminal window is named LBBB;
a real R' (RBBB), a short intrinsicoid deflection (anterior STEMI widening the QRS) or a narrower QRS keep RBBB."""
import pytest

from cardiolens.contracts.features import (
    Intervals,
    Measurements,
    Morphology,
    STMeasurement,
    Voltages,
)
from cardiolens.layer4_findings.rules import _lbbb_read_as_rbbb


def _m(qrs=170.0, idt=90.0, v1_net=-2000.0, v1_r=500.0, v1_j=500.0, retry=False):
    m = Measurements(intervals=Intervals(qrs_ms=qrs), voltages=Voltages(intrinsicoid_v5_v6_ms=idt),
                     morphology=Morphology(st=[STMeasurement(lead="V1", qrs_net_uv=v1_net, r_height_uv=v1_r,
                                                             j_point_uv=v1_j, s_depth_uv=2000.0)]))
    m.qrs_width_from_retry = retry
    return m


def test_lbbb_with_v1_st_elevation_is_lbbb():
    assert _lbbb_read_as_rbbb(_m())


@pytest.mark.parametrize("kw", [dict(idt=40.0),               # anterior STEMI widening the QRS: early V5-V6 R peak
                                dict(qrs=125.0),              # not wide enough
                                dict(v1_net=900.0),           # V1 positive: a real RBBB
                                dict(v1_r=900.0, v1_j=100.0),  # an R' well above J: a real RBBB
                                dict(retry=True)])            # width from the retry search
def test_rbbb_kept(kw):
    assert not _lbbb_read_as_rbbb(_m(**kw))


def test_missing_measurements_keep_rbbb():
    m = _m()
    m.morphology.st = []
    assert not _lbbb_read_as_rbbb(m)
    assert not _lbbb_read_as_rbbb(_m(idt=None))
