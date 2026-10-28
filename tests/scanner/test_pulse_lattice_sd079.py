"""SD-079: when the 5-mm grid's lines get into the ink mask (a dark grid, a compressed copy),
its cells are not taken for calibration pulses, and the widened retry is not run there."""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.calibration.pulse import find_pulses, grid_in_mask, ink_mask, summarize_pulses
from ecgscan.layout import get_layout

pytest.importorskip("cv2")


def _page(mv, *, gain: int, speed: int = 25, lines: int | None = None, shift_mm: int = 2):
    page, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"),
                          RenderSpec(px_per_mm=10.0, gain_mm_mv=gain, speed_mm_s=speed, grid="gray"))
    img = page.copy()
    if lines is not None:                              # 5-mm lines dark enough for the ink mask
        h, w = img.shape[:2]
        for x in range(shift_mm * 10, w, 50):
            img[:, x:x + 2] = np.minimum(img[:, x:x + 2], lines)
        for y in range(0, h, 50):
            img[y:y + 2] = np.minimum(img[y:y + 2], lines)
    return img


@pytest.mark.parametrize("gain", [10, 20])
def test_grid_cells_in_the_mask_are_not_pulses(synth_mv, gain):
    """Before SD-079 the cells (4-5 mm) won with 5-9 agreeing: gain 5 for a 10 or 20 mm/mV page."""
    ink = ink_mask(_page(synth_mv, gain=gain, lines=130), 10.0)
    assert grid_in_mask(ink, 10.0)
    s = summarize_pulses(find_pulses(ink, 10.0))
    assert s is None or s["height_mm"] == pytest.approx(gain, rel=0.08)


@pytest.mark.parametrize("gain", [5, 10, 20])
def test_a_page_without_grid_lines_in_the_mask_is_unchanged(synth_mv, gain):
    ink = ink_mask(_page(synth_mv, gain=gain), 10.0)
    assert not grid_in_mask(ink, 10.0)
    s = summarize_pulses(find_pulses(ink, 10.0))
    assert s["n_agree"] == 4 and s["height_mm"] == pytest.approx(gain, rel=0.08)


# --- SD-084: a pulse drawn on the grid's lines (ECG-Image-Kit) -------------------------

def _kit_like(*, pulse_grey: int, line_grey: int = 76, ppm: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """A grey 5-mm grid in the mask and an open 10 x 5 mm pulse on two of its lines, its legs
    stopping 1.5 mm above the trace (the kit's clean render: legs grey 29, lines 76)."""
    h, w = 300, 400
    g = np.full((h, w), 250, np.uint8)
    for x in range(0, w, 50):
        g[:, x:x + 3] = line_grey
    for y in range(25, h, 50):
        g[y:y + 3] = line_grey
    top, foot = 80, 180
    g[top:foot, 100:104] = pulse_grey
    g[top:foot, 150:154] = pulse_grey
    g[top:top + 4, 100:154] = pulse_grey
    g[195:198, 156:w] = 0                                 # the trace, after the pulse
    return g < 120, g


def test_a_pulse_drawn_on_the_grid_lines_is_a_pulse():
    """Before SD-084 its two legs, on 5-mm lines, were taken for a grid cell: no gain."""
    mask, g = _kit_like(pulse_grey=29)
    assert grid_in_mask(mask, 10.0)
    assert find_pulses(mask, 10.0) == []                  # without the page's grey: a cell
    (p,) = find_pulses(mask, 10.0, gray=g)
    assert p.height_mm == pytest.approx(10.0, abs=0.3) and p.width_mm == pytest.approx(5.0, abs=0.3)


def test_a_grid_cell_as_dark_as_its_lines_stays_a_cell():
    mask, g = _kit_like(pulse_grey=76)
    assert find_pulses(mask, 10.0, gray=g) == []
