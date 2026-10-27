"""SD-075: a board from an office printer. The paper feed is uneven (the owner's print: ~1 mm off
at the start, 2-3 mm in the middle, back on at the end), so the board gives the plane and the
paper grid the ECG's scale."""

import copy

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.board import detect_board, load_boards, render_board
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, _board, pipeline_config, scan

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

TILT = np.float32([[120, 80], [0.95, 30], [1.0, 0.98], [60, 0.9]])     # as in test_board.py

FEEDS = {    # x_printed - x_board in mm along the feed (the board's x), W = board width
    "owner, 101 %": lambda x, W: 0.01 * x + 1.0 * (1 - x / W) + 2.0 * np.sin(np.pi * x / W),
    "99 %, sine 1.5 mm": lambda x, W: -0.01 * x + 1.5 * np.sin(2 * np.pi * x / W),
    "owner x2": lambda x, W: 2.0 * (1 - x / W) + 4.0 * np.sin(np.pi * x / W),   # middle markers > 1 mm off
}


@pytest.fixture(scope="module")
def a3():
    return load_boards(["a3", "a4"])[0]


def _feed(img: np.ndarray, ppm: float, err, width_mm: float) -> np.ndarray:
    """``img`` printed with the feed error ``err``: what is at board x lands at x + err(x)."""
    h, w = img.shape[:2]
    xb = np.arange(2 * w) / 2 / ppm
    src = np.interp(np.arange(w) / ppm, xb + err(xb, width_mm), xb) * ppm
    mapx = np.tile(src.astype(np.float32), (h, 1))
    mapy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderValue=255)


def _photo(mv, spec, err, *, bent: bool = False, page: bool = True, ppm: float = 12.0) -> np.ndarray:
    """A true-scale 3x4 page on a board printed with ``err``, photographed at an angle.
    ``bent``: the board is printed true but lifted, the page on it stretched alike."""
    board = render_board(spec, ppm)
    if not bent:
        board = _feed(board, ppm, err, spec.size_mm[0])
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    if page:
        pg, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=ppm))
        y0 = x0 = int(50 * ppm)
        board[y0:y0 + pg.shape[0], x0:x0 + pg.shape[1]] = pg
    if bent:
        board = _feed(board, ppm, err, spec.size_mm[0])
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [w * TILT[1, 0], TILT[1, 1]], [w * TILT[2, 0], h * TILT[2, 1]],
                      [TILT[3, 0], h * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(w * 1.05), int(h * 1.02)),
                               borderValue=(90, 80, 70))


def _median_r(mv, res) -> float:
    rs = []
    for lead, tr in res.leads.items():
        y = tr.array()
        k = int(round(tr.start_s * 500))
        t = mv[lead][k:k + len(y)]
        ok = np.isfinite(y[:len(t)])
        rs.append(np.corrcoef(t[ok], y[:len(t)][ok])[0, 1])
    return float(np.median(rs))


def test_the_corner_markers_of_each_board(a3):
    a4 = load_boards(["a3", "a4"])[1]
    assert a3.corner_ids == [0, 2, 4, 6] and a4.corner_ids == [10, 12, 14, 16]


@pytest.mark.parametrize("feed", ["owner, 101 %", "99 %, sine 1.5 mm"])
def test_an_unevenly_printed_board_reads_the_page_at_its_own_scale(synth_mv, a3, feed):
    """Before SD-075 the board's 1 % went into the timing: the zone's grid read 9.92 / 10.12
    px/mm and the leads' r fell to 0.26 / 0.03 (0.90 on a true board)."""
    dbg = ScanDebug()
    res = scan(cv2.imencode(".png", _photo(synth_mv, a3, FEEDS[feed]))[1].tobytes(), debug=dbg)
    assert dbg.rectified.method == "board:a3" and not dbg.board.corners_only
    g = dbg.rectified.grid
    assert g.px_per_mm_x == pytest.approx(10.0, rel=0.001) and g.px_per_mm_y == pytest.approx(10.0, rel=0.001)
    assert res.status in ("ok", "needs_confirmation") and not res.reject_reasons
    assert _median_r(synth_mv, res) > 0.85


def test_middle_markers_far_off_the_plane_is_taken_from_the_corners(synth_mv, a3):
    """Middle markers ~3.4 mm off: the 1-mm residual refused the board (and the grid path read
    QRS 114 ms for 66). The corners give the plane, the even paper grid the scale."""
    photo = _photo(synth_mv, a3, FEEDS["owner x2"])
    fit = detect_board(photo, load_boards(["a3", "a4"]))
    assert fit.ok and fit.corners_only and fit.residual_mm < 1.0 and "uneven print" in fit.notes[0]
    dbg = ScanDebug()
    rect, _, _ = _board(photo, copy.deepcopy(pipeline_config()), dbg)
    assert rect.method == "board:a3" and rect.grid.px_per_mm_x == pytest.approx(10.0, rel=0.001)


def test_a_bent_board_is_not_taken_for_an_uneven_print(synth_mv, a3):
    """The same middle-marker offsets, but the page on the board is stretched with it: the
    paper grid is uneven over the page, so the corner fit is not trusted (the grid path then)."""
    dbg = ScanDebug()
    assert _board(_photo(synth_mv, a3, FEEDS["owner x2"], bent=True), copy.deepcopy(pipeline_config()), dbg) is None
    assert dbg.board.corners_only and not dbg.board.ok and "uneven over the page" in dbg.board.reason


def test_a_corner_fit_needs_a_paper_grid_to_check_it(synth_mv, a3):
    dbg = ScanDebug()
    photo = _photo(synth_mv, a3, FEEDS["owner x2"], page=False)
    assert _board(photo, copy.deepcopy(pipeline_config()), dbg) is None
    assert dbg.board.corners_only and "without a paper grid" in dbg.board.reason


# --- SD-076: the A3 board printed on two A4 sheets -------------------------------------

def test_the_two_sheet_board_holds_an_a4_page_and_prints_on_a4(synth_mv):
    from ecgscan.board import load_board, render_sheets

    spec = load_board("a3x2")
    zw, zh = spec.zone_size_mm
    assert zw >= 297 + 10 and zh >= 210 + 5                         # an A4 page landscape, with room
    assert spec.corner_ids == [20, 22, 24, 26] and len(spec.markers) == 6
    sheets = render_sheets(spec, 10.0)
    assert [s.shape for s in sheets] == [(2970, 2100), (2970, 2100)]  # A4 portrait each
    cut = int(spec.cut_mm * 10)
    for mid, (cx, _) in spec.markers.items():                        # no marker crosses the seam
        assert abs(cx * 10 - cut) > (spec.marker_mm / 2 + 10) * 10, mid


def _taped(mv, ppm: float, shift_mm=(0.8, -0.6), turn_deg: float = 0.3) -> np.ndarray:
    """Sheet 2 on the table, sheet 1 cut along its line and laid over it a little off."""
    from ecgscan.board import load_board, render_sheets

    spec = load_board("a3x2")
    s1, s2 = render_sheets(spec, ppm)
    w, h = int(round(spec.size_mm[0] * ppm)), s1.shape[0]
    board = np.full((h, w), 255, np.uint8)
    a2 = int(round(spec.sheets[1][0] * ppm))
    m = cv2.getRotationMatrix2D((0, h / 2), turn_deg, 1.0)
    m[:, 2] += [a2 + shift_mm[0] * ppm, shift_mm[1] * ppm]
    placed = cv2.warpAffine(s2, m, (w, h), borderValue=255)
    board = np.minimum(board, placed)
    cut = int(round(spec.cut_mm * ppm))
    board[:, :cut] = s1[:, :cut]
    board = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    pg, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=ppm))
    y0 = x0 = int(45 * ppm)
    board[y0:y0 + pg.shape[0], x0:x0 + pg.shape[1]] = pg
    hh, ww = board.shape[:2]
    src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [ww * TILT[1, 0], TILT[1, 1]], [ww * TILT[2, 0], hh * TILT[2, 1]],
                      [TILT[3, 0], hh * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(ww * 1.05), int(hh * 1.02)),
                               borderValue=(90, 80, 70))


def test_a_page_on_two_taped_sheets_is_read_at_its_own_scale(synth_mv):
    """Sheet 1 laid 0.8 mm / 0.3 deg off sheet 2: the board gives the plane, the paper the scale."""
    dbg = ScanDebug()
    res = scan(cv2.imencode(".png", _taped(synth_mv, 12.0))[1].tobytes(), debug=dbg)
    assert dbg.rectified.method == "board:a3x2" and dbg.board.residual_mm < 1.0
    assert dbg.rectified.grid.px_per_mm_x == pytest.approx(10.0, rel=0.001)
    assert res.status in ("ok", "needs_confirmation") and not res.reject_reasons
    assert _median_r(synth_mv, res) > 0.85
