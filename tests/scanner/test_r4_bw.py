"""R4 (SD-104): a black grid whose spacing is not exactly 5 mm at the measured scale (a scan's
scale off by a little, a page curled in the scanner) is still taken out of the trace mask."""

import numpy as np

from ecgscan.segmentation.grid_lines import drop_grid_lines

PPM = 10.0


def _drifting_grid(h=2200, w=1200, spacing=50.6):
    """Horizontal grid lines 1.2 % wider apart than 5 mm at PPM: off by a quarter period after
    ~20 lines. One wavy trace in the middle."""
    ink = np.zeros((h, w), bool)
    y = 0.0
    while y < h - 2:
        ink[int(y):int(y) + 2, :] = True
        y += spacing
    xs = np.arange(w)
    ys = (1100 + 18 * np.sin(xs / 30.0)).astype(int)
    for x, yy in zip(xs, ys, strict=True):
        ink[yy - 1:yy + 2, x] = True
    return ink, ys


def test_per_tile_the_drifting_lines_go_and_the_trace_stays():
    ink, ys = _drifting_grid()
    whole, _ = drop_grid_lines(ink, PPM)
    tiled, dropped = drop_grid_lines(ink, PPM, tile_mm=20.0)
    left = lambda out: out[:900].mean() + out[1300:].mean()       # away from the trace
    assert left(tiled) < 0.01 < left(whole)                       # one phase per page leaves lines
    assert dropped > 0.5
    kept = np.mean([tiled[y - 1:y + 2, x].any() for x, y in enumerate(ys)])
    assert kept >= 0.97
