"""R3 (SD-103): the short limb-lead windows of a 3x4 page. Where the DP path has a gap the D47
tracker may fill it, but only within what a QRS can do in that many columns."""

import numpy as np
import pytest

from ecgscan.vectorize import _max_step_px, fuse

PITCH = 300.0


def _pair(gap_values):
    path = np.full(60, 200.0)
    detail = path.copy()
    path[30:30 + len(gap_values)] = np.nan
    detail[30:30 + len(gap_values)] = gap_values
    return path, detail


def test_a_one_column_jump_into_a_path_gap_stays_a_gap():
    path, detail = _pair([343.0])                    # 18 mm off a flat line at ~8 px/mm
    out = fuse(path, detail, PITCH, max_step_px=80.0)
    assert np.isnan(out[30])
    assert np.isfinite(out[:30]).all() and np.isfinite(out[31:]).all()


def test_a_steep_edge_or_a_near_apex_in_a_path_gap_is_kept():
    path, detail = _pair([180.0, 120.0, 90.0, 140.0])  # an apex 110 px up over 4 columns
    path[34:] = 150.0
    detail[34:] = 150.0
    out = fuse(path, detail, PITCH, max_step_px=80.0)
    assert np.allclose(out[30:34], [180.0, 120.0, 90.0, 140.0])


def test_without_the_limit_the_old_behaviour_holds():
    path, detail = _pair([343.0])
    assert fuse(path, detail, PITCH)[30] == 343.0


def test_the_step_follows_speed_and_gain_not_resolution():
    cfg = {"fuse_gap_max_slope_mv_ms": 0.2}
    assert _max_step_px(cfg, 25.0, 10.0, 8.0) == pytest.approx(80.0)
    assert _max_step_px(cfg, 25.0, 10.0, 12.0) == pytest.approx(80.0)
    assert _max_step_px(cfg, 50.0, 10.0, 8.0) == pytest.approx(40.0)     # a column is half as long in time
    assert _max_step_px({}, 25.0, 10.0, 8.0) is None


def test_the_strip_is_suspect_only_when_every_reference_disagrees():
    from ecgscan.validate import rhythm_inconsistent

    assert not rhythm_inconsistent({"panel_II": 0.77, "aVF_aVR": 0.08}, 0.5, "all")   # panel II misread
    assert rhythm_inconsistent({"panel_II": 0.74, "aVF_aVR": 0.57}, 0.5, "all")       # the strip itself
    assert rhythm_inconsistent({"panel_II": 0.77}, 0.5, "all")                        # its only reference
    assert rhythm_inconsistent({"panel_II": 0.77, "aVF_aVR": 0.08}, 0.5, "any")       # the old rule
    assert not rhythm_inconsistent({}, 0.5, "all")
