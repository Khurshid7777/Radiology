"""SD-077: paper laid over the board's rulers, beyond the zone, is read up to the markers; the
board's own print (rulers, zone outline, title) is never read."""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.board import load_board, render_board
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, scan

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

TILT = np.float32([[120, 80], [0.95, 30], [1.0, 0.98], [60, 0.9]])     # as in test_board.py
PPM = 12.0


def _photo(mv, *, top_mm: float | None, grid: str = "red") -> np.ndarray:
    """A3 board; a 6x2 page cropped to 274 x 216 mm (rows near its edges) at ``top_mm``.
    The zone is 217 mm high from y = 40: at 40.5 the page is inside, at 49.5 its bottom row
    runs over the ruler (the paper ends 1.5 mm short of the markers)."""
    spec = load_board("a3")
    board = cv2.cvtColor(render_board(spec, PPM), cv2.COLOR_GRAY2BGR)
    if top_mm is not None:
        pg, _ = render_page(mv, 500.0, get_layout("6x2"), RenderSpec(px_per_mm=PPM, row_pitch_mm=33, grid=grid))
        pg = pg[int(2 * PPM):pg.shape[0] - int(16 * PPM)]
        x, y = int(55 * PPM), int(top_mm * PPM)
        board[y:y + pg.shape[0], x:x + pg.shape[1]] = pg
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [w * TILT[1, 0], TILT[1, 1]], [w * TILT[2, 0], h * TILT[2, 1]],
                      [TILT[3, 0], h * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(w * 1.05), int(h * 1.02)),
                               borderValue=(90, 80, 70))


def _scan(img):
    dbg = ScanDebug()
    return scan(cv2.imencode(".png", img)[1].tobytes(), debug=dbg), dbg


def _r(mv, res) -> dict[str, float]:
    out = {}
    for lead, tr in res.leads.items():
        y = tr.array()
        k = int(round(tr.start_s * 500))
        t = mv[lead][k:k + len(y)]
        ok = np.isfinite(y[:len(t)])
        out[lead] = float(np.corrcoef(t[ok], y[:len(t)][ok])[0, 1])
    return out


def test_the_read_area_reaches_the_markers_inner_edges():
    a4 = load_board("a4")
    x0, y0, x1, y1 = a4.read_mm(1.5)
    assert (x1 - x0, y1 - y0) == (248.0, 161.0)            # zone 231 x 144: a 235 x 148 strip fits
    for cx, cy in a4.markers.values():                    # no marker reaches into it
        h = a4.marker_mm / 2
        assert cx + h <= x0 or cx - h >= x1 or cy + h <= y0 or cy - h >= y1


def test_paper_over_the_ruler_is_read_whole(synth_mv):
    """Before SD-077 the zone cut the bottom row: needs_confirmation (LEAD_INCOMPLETE,
    LEAD_OVERLAP, TRACE_GAPS), V6 r 0.825 for 0.891 inside the zone."""
    res, dbg = _scan(_photo(synth_mv, top_mm=49.5))
    assert dbg.rectified.method == "board:a3" and res.status == "ok" and res.layout.name == "6x2"
    assert dbg.rectified.bgr.shape[0] > 217 * 10                       # beyond the zone's height
    r = _r(synth_mv, res)
    assert len(r) == 12 and min(r.values()) > 0.85 and r["V6"] > 0.87


def test_the_bare_board_print_is_never_read(synth_mv):
    res, _ = _scan(_photo(synth_mv, top_mm=None))
    assert res.status == "rejected" and not res.leads


def test_grey_grid_paper_is_read_inside_the_zone_as_before(synth_mv):
    """Untinted paper is not told from the board's grey print: the zone alone is read."""
    _, dbg = _scan(_photo(synth_mv, top_mm=49.5, grid="gray"))
    assert dbg.rectified.method == "board:a3" and dbg.rectified.bgr.shape[0] <= 217 * 10
