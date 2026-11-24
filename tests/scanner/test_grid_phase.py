"""SD-143: the grid's phase, period and zone agreement in FrameQC, on synthetic frames with a known grid."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ecgscan.capture_qc import check_frame, phase_line, refine_period

spec = importlib.util.spec_from_file_location("grid_phase_check",
                                              Path(__file__).parents[2] / "tools" / "scanner" / "grid_phase_check.py")
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)


def shot(scale_ppm, rot=0.0, persp=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pg = gp.page(200 + seed, "red")
    s = scale_ppm / gp.PAGE_PPM
    ox, oy = (gp.FW - pg.shape[1] * s) / 2 + 7, (gp.FH - pg.shape[0] * s) / 2 - 5
    bgr, H = gp.frame(pg, s, ox, oy, rot, persp, rng)
    return check_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), H


@pytest.mark.parametrize("ppm", [5.0, 6.0])  # where the period itself is found right (the doubling is SD-140)
def test_a_flat_frame_reports_where_the_lines_are(ppm):
    q, H = shot(ppm)
    assert q.grid_period_px == pytest.approx(5 * ppm, rel=0.01)
    assert gp.phase_error(q, H, 0) <= 0.1 and gp.phase_error(q, H, 1) <= 0.1
    assert q.grid_zones_agree >= 0.9 and q.grid_zones == 9
    assert 0 <= q.grid_phase[0] < 1 and 0 <= q.grid_phase[1] < 1
    assert q.grid_phase_px[0] == pytest.approx(q.grid_phase[0] * q.grid_period_px, abs=0.05)


def test_a_turned_sheet_is_followed_by_the_slope():
    q, H = shot(5.0, rot=3.0, seed=1)
    assert gp.phase_error(q, H, 0) <= 0.1 and gp.phase_error(q, H, 1) <= 0.1
    assert q.grid_zones_agree >= 0.9
    assert abs(q.grid_slope[0]) == pytest.approx(np.tan(np.radians(3.0)), abs=0.01)


def test_a_tilted_camera_breaks_the_agreement():
    q, _ = shot(5.0, persp=0.25, seed=2)
    assert q.grid_period_px is not None and q.grid_zones_agree <= 0.6


def test_no_grid_no_geometry():
    q = check_frame(np.full((720, 1280, 3), 200, np.uint8))
    assert q.grid_phase_px is q.grid_phase is q.grid_slope is q.grid_period_px is q.grid_zones_agree is q.grid_zones is None


def test_phase_and_period_on_a_pure_profile():
    x = np.arange(1280)
    prof = np.tile(np.cos(2 * np.pi * (x - 7.0) / 23.4), (6, 1))
    assert refine_period(prof, 23.0) == pytest.approx(23.4, abs=0.05)
    phase, slope = phase_line(prof, 23.4, band=16)
    assert phase == pytest.approx(7.0, abs=0.2) and slope == pytest.approx(0.0, abs=1e-3)
