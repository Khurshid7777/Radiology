"""R1 (D185 real images): what a 200-dpi scan and a clean render needed — a black 5-mm grid
taken out of the mask (SD-099), column-separator bars read as missing (SD-100), light-red
grid pixels never trace (SD-101)."""

import numpy as np

from ecgscan.segmentation.grid_lines import drop_grid_lines, drop_red
from ecgscan.vectorize import separator_span

PPM = 10.0


def _page(h=600, w=1200, tilt_px=0):
    """A black 5-mm grid (2 px lines) and one wavy trace across it."""
    ink = np.zeros((h, w), bool)
    for y in range(0, h, 50):
        for x in range(w):
            yy = y + (tilt_px * x) // w
            if 0 <= yy < h - 1:
                ink[yy:yy + 2, x] = True
    ink[:, ::50] = True
    ink[:, 1::50] = True
    xs = np.arange(w)
    ys = (300 + 18 * np.sin(xs / 30.0)).astype(int)
    for x, y in zip(xs, ys, strict=True):
        ink[y - 1:y + 2, x] = True
    return ink, ys


def test_the_grid_goes_and_the_trace_stays():
    for tilt in (0, 4):                       # a scan a fraction of a degree off square
        ink, ys = _page(tilt_px=tilt)
        out, dropped = drop_grid_lines(ink, PPM)
        assert dropped > 0.5
        # The trace keeps its ink where it crosses a line; only where it runs along one for a
        # few columns (the same pixels) is it lost: a gap, never a wrong value.
        kept = np.mean([out[y - 1:y + 2, x].any() for x, y in enumerate(ys)])
        assert kept >= 0.97
        # Off the trace (rows 100-200) almost nothing of the grid is left.
        assert out[100:200].mean() < 0.01


def test_a_page_without_a_grid_in_the_mask_is_unchanged():
    ink = np.zeros((600, 1200), bool)
    xs = np.arange(1200)
    ink[(300 + 18 * np.sin(xs / 30.0)).astype(int), xs] = True
    out, dropped = drop_grid_lines(ink, PPM)
    assert dropped == 0.0 and np.array_equal(out, ink)


def test_a_separator_bar_is_found_and_a_qrs_is_not():
    ink = np.zeros((400, 800), bool)
    ink[170:230, 398:404] = True              # a 6-mm bar at the column boundary x=400
    assert separator_span(ink, 400, 100, 300, PPM) == (398, 403)
    qrs = np.zeros((400, 800), bool)
    for i, x in enumerate(range(395, 405)):   # an R wave: 10 mm over 10 columns, ~1 mm per column
        qrs[200 - 10 * (i + 1):200 - 10 * i, x] = True
    assert separator_span(qrs, 400, 100, 300, PPM) is None


def test_light_red_goes_dark_ink_stays():
    bgr = np.full((10, 4, 3), 255, np.uint8)
    bgr[:, 0] = (60, 60, 230)                 # a light-red grid line (BGR)
    bgr[:, 1] = (20, 20, 20)                  # the black trace
    bgr[:, 2] = (40, 50, 90)                  # the trace under a warm lamp: tinted, but dark
    bgr[:, 3] = (150, 160, 200)               # pale pink paper the mask wrongly kept
    ink = np.ones((10, 4), bool)
    out = drop_red(ink, bgr)
    assert out[:, 1].all() and out[:, 2].all() and not out[:, 0].any()
    assert out[:, 3].all()                    # only 40 levels redder: not a grid line
    part = np.ascontiguousarray(ink[:, 1:3])
    assert drop_red(part, bgr[:, 1:3]) is part      # nothing red: the mask itself, untouched
