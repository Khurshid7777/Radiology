"""Synthetic pages draw exactly what the truth says; defects are seeded and mapped."""

import numpy as np
import pytest

from ecgscan.bench.degrade import DegradeSpec, degrade
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.layout import get_layout

cv2 = pytest.importorskip("cv2")


def _ink_rows(img, x: float) -> np.ndarray:
    col = img[:, int(round(x))].astype(int)
    return np.flatnonzero(col.max(axis=1) < 120)          # trace is dark grey on light paper/grid


@pytest.mark.parametrize("ppm,speed,gain", [(7.5, 25, 10), (10.0, 50, 20), (6.0, 25, 5)])
def test_trace_is_drawn_where_the_truth_says(synth_mv, ppm, speed, gain):
    img, gt = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"),
                          RenderSpec(px_per_mm=ppm, speed_mm_s=speed, gain_mm_mv=gain, labels=False, header=False))
    rhythm = next(t for t in gt.traces if t.kind == "rhythm")
    errs = []
    for k in range(300, 4700, 211):                       # skip pulse area / ends
        x = rhythm.x0_px + k / gt.fs * speed * ppm
        y = rhythm.baseline_px - rhythm.samples_mv[k] * gain * ppm
        rows = _ink_rows(img, x)
        assert rows.size, (ppm, k)
        errs.append(np.min(np.abs(rows - y)))
    assert np.median(errs) <= 1.0                          # within a pixel of the true point


def test_page_width_scales_with_speed_and_truth_covers_layout(synth_mv):
    lay = get_layout("3x4+rhythm")
    a, gta = render_page(synth_mv, 500.0, lay, RenderSpec(px_per_mm=6, speed_mm_s=25))
    b, _ = render_page(synth_mv, 500.0, lay, RenderSpec(px_per_mm=6, speed_mm_s=50))
    assert b.shape[1] - a.shape[1] == pytest.approx(lay.record_s * 25 * 6, abs=2)
    assert sorted((t.lead, t.kind) for t in gta.traces) == sorted(
        [(ld, "panel") for ld in lay.leads()] + [("II", "rhythm")])
    assert all(len(t.samples_mv) == (5000 if t.kind == "rhythm" else 1250) for t in gta.traces)


def test_short_signal_is_refused(synth_mv):
    short = {k: v[:3000] for k, v in synth_mv.items()}
    with pytest.raises(ValueError):
        render_page(short, 500.0, get_layout("12x1"), RenderSpec(px_per_mm=4))


def test_degrade_is_seeded_and_maps_the_page(synth_mv):
    img, _ = render_page(synth_mv, 500.0, get_layout("single"), RenderSpec(px_per_mm=5))
    spec = DegradeSpec(background=0.05, perspective=0.03, rotation_deg=4, noise=3, stains=1, scale=0.5)
    a, info = degrade(img, spec, seed=11, px_per_mm=5)
    b, _ = degrade(img, spec, seed=11, px_per_mm=5)
    c, _ = degrade(img, spec, seed=12, px_per_mm=5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    quad = np.array(info["page_quad_px"])
    assert quad.min() >= -1 and quad[:, 0].max() <= a.shape[1] + 1 and quad[:, 1].max() <= a.shape[0] + 1
    assert 0.4 < info["centre_scale"] < 0.6 and not info["nonrigid"]


def test_noop_degrade_keeps_the_page(synth_mv):
    img, _ = render_page(synth_mv, 500.0, get_layout("single"), RenderSpec(px_per_mm=4))
    out, info = degrade(img, DegradeSpec(), seed=0, px_per_mm=4)
    assert np.array_equal(out, img) and info["centre_scale"] == pytest.approx(1.0)
