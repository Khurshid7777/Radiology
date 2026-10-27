"""SD-065: the capture board gives the exact scale, the photo's resolution and the zone."""

import copy

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.board import detect_board, load_boards, render_board
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, _board, _board_set, pipeline_config, scan

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

TILT = np.float32([[120, 80], [0.95, 30], [1.0, 0.98], [60, 0.9]])     # px, or a fraction of w / h


@pytest.fixture(scope="module")
def boards():
    return load_boards(["a3", "a4"])


def _photo(mv, spec, ppm: float, *, page_scale: float = 1.0, tilt: bool = True, hide: tuple[int, ...] = ()):
    """The board at ``ppm`` with a 3x4 page on it (printed at ``page_scale``), photographed at an angle."""
    board = cv2.cvtColor(render_board(spec, ppm), cv2.COLOR_GRAY2BGR)
    page, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=ppm * page_scale))
    x0, y0 = int(50 * ppm), int(50 * ppm)
    board[y0:y0 + page.shape[0], x0:x0 + page.shape[1]] = page
    for mid in hide:                                       # a hand or a fold over a marker
        cx, cy = spec.markers[mid]
        r = int(spec.marker_mm * ppm)
        board[max(0, int(cy * ppm) - r):int(cy * ppm) + r, max(0, int(cx * ppm) - r):int(cx * ppm) + r] = 200
    if not tilt:
        return board
    h, w = board.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[TILT[0, 0], TILT[0, 1]], [w * TILT[1, 0], TILT[1, 1]], [w * TILT[2, 0], h * TILT[2, 1]],
                      [TILT[3, 0], h * TILT[3, 1]]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (int(w * 1.05), int(h * 1.02)),
                               borderValue=(90, 80, 70))


def test_marker_ids_are_unique_across_boards():
    with pytest.raises(ValueError, match="both"):
        load_boards(["a3", "a3"])


def test_flat_board_maps_to_millimetres_exactly(boards):
    a4 = boards[1]
    img = cv2.cvtColor(render_board(a4, 8.0), cv2.COLOR_GRAY2BGR)
    fit = detect_board(img, boards)
    assert fit.ok and fit.spec.name == "a4" and fit.marker_ids == sorted(a4.markers)
    assert fit.residual_mm < 0.1
    assert fit.min_px_per_mm == pytest.approx(8.0, rel=0.01) and fit.mean_px_per_mm == pytest.approx(8.0, rel=0.01)


def test_no_board_is_none_and_the_grid_path_is_taken(synth_mv):
    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=8))
    assert detect_board(page, load_boards(["a3", "a4"])) is None


def test_tilted_board_page_is_read_at_exact_scale(synth_mv, boards):
    """At this tilt the grid path loses the traces (r < 0.35); the board reads them."""
    photo = _photo(synth_mv, boards[0], 12.0)
    dbg = ScanDebug()
    res = scan(cv2.imencode(".png", photo)[1].tobytes(), debug=dbg)
    assert dbg.rectified.method == "board:a3" and dbg.board.residual_mm < 0.1
    assert dbg.rectified.grid.px_per_mm == pytest.approx(10.0, rel=0.002)        # the paper grid agrees
    assert res.status in ("ok", "needs_confirmation") and not res.reject_reasons
    assert 9.5 < res.quality.px_per_mm < 11                # the photo's worst point of the zone
    rs = []
    for lead, tr in res.leads.items():
        y = tr.array()
        k = int(round(tr.start_s * 500))
        t = synth_mv[lead][k:k + len(y)]
        ok = np.isfinite(y[:len(t)])
        rs.append(np.corrcoef(t[ok], y[:len(t)][ok])[0, 1])
    assert len(rs) == 12 and np.median(rs) > 0.85


def test_low_resolution_is_judged_at_the_worst_point_of_the_zone(synth_mv, boards):
    dbg = ScanDebug()
    res = scan(cv2.imencode(".png", _photo(synth_mv, boards[0], 9.5))[1].tobytes(), debug=dbg)
    assert res.reject_reasons == ["LOW_RESOLUTION"] and dbg.board.ok
    assert res.quality.px_per_mm < 8 < dbg.board.mean_px_per_mm     # an average would have passed it


@pytest.mark.parametrize(("kw", "reason"), [
    ({"hide": (0, 1, 2, 3, 4, 5)}, "markers seen"),                        # 2 left: not trusted
    ({"page_scale": 0.95, "tilt": False}, "paper grid"),                   # printed at 95 %: the paper wins
])
def test_an_untrusted_board_falls_back_to_the_grid(synth_mv, boards, kw, reason):
    cfg = copy.deepcopy(pipeline_config())
    dbg = ScanDebug()
    assert _board(_photo(synth_mv, boards[0], 12.0, **kw), cfg, dbg) is None
    assert not dbg.board.ok and reason in dbg.board.reason


# --- a page in parts (SD-066) ---------------------------------------------------------

def _shoot(img, scale: float, tilt: int = 60):
    hh, ww = img.shape[:2]
    src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    dst = np.float32([[tilt, tilt], [ww - tilt, 0], [ww, hh], [0, hh - tilt]]) * scale
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (int(ww * scale), int(hh * scale)),
                               borderValue=(90, 80, 70))


@pytest.fixture(scope="module")
def a3_parts(synth_mv, boards):
    """The whole A3 at ~7 px/mm (refused), and its two halves at ~11 px/mm."""
    flat = _photo(synth_mv, boards[0], 12.0, tilt=False)
    w = flat.shape[1]
    return _shoot(flat, 0.6), _shoot(flat[:, :int(w * 0.56)], 0.92), _shoot(flat[:, int(w * 0.44):], 0.92)


def _png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def test_a_page_too_coarse_whole_is_read_from_two_halves(synth_mv, a3_parts):
    whole, left, right = a3_parts
    assert scan(_png(whole)).reject_reasons == ["LOW_RESOLUTION"]
    dbg = ScanDebug()
    res = scan([_png(left), _png(right)], debug=dbg)
    assert not res.reject_reasons and "STITCHED" in res.warnings
    assert dbg.composite.coverage == pytest.approx(1.0) and set(np.unique(dbg.composite.source)) == {0, 1}
    assert res.quality.px_per_mm > 9 and len(res.source.parts_sha256) == 2
    rs = []
    for lead, tr in res.leads.items():
        y = tr.array()
        k = int(round(tr.start_s * 500))
        t = synth_mv[lead][k:k + len(y)]
        ok = np.isfinite(y[:len(t)])
        rs.append(np.corrcoef(t[ok], y[:len(t)][ok])[0, 1])
    assert len(rs) == 12 and np.median(rs) > 0.85


def test_a_set_needs_markers_on_every_part_and_the_whole_zone(a3_parts, synth_mv):
    left = a3_parts[1]
    cfg = copy.deepcopy(pipeline_config())
    plain, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=8))
    assert _board_set([left, plain], cfg, ScanDebug()) == "BOARD_NOT_FOUND"
    assert _board_set([left, left], cfg, ScanDebug()) == "PAGE_INCOMPLETE"       # the right half never seen
    res = scan([_png(left), _png(plain)])
    assert res.reject_reasons == ["BOARD_NOT_FOUND"] and res.source.parts_sha256


def test_wide_a4_board_holds_a_152_mm_strip_and_is_told_from_a4():
    """D176: a 148-mm thermal strip overflowed the a4 zone (144 mm); a4w (ids 20-27, 14-mm
    markers, 3-mm ruler ticks) holds 152 mm and an a4 print keeps reading as a4."""
    specs = load_boards(["a3", "a4", "a4w"])
    a4w = specs[2]
    assert a4w.zone_size_mm[1] >= 150
    img = cv2.cvtColor(render_board(a4w, 8.0), cv2.COLOR_GRAY2BGR)
    fit = detect_board(img, specs)
    assert fit.ok and fit.spec.name == "a4w"
    old = cv2.cvtColor(render_board(specs[1], 8.0), cv2.COLOR_GRAY2BGR)
    assert detect_board(old, specs).spec.name == "a4"
    # >= 3 mm of white between every marker and the rulers (the ArUco quiet zone).
    x0, y0, _x1, _y1 = a4w.zone_mm
    h = a4w.marker_mm / 2
    assert y0 - 1 - a4w.ruler_mm - (15 + h) >= 3 and x0 - 1 - a4w.ruler_mm - (15 + h) >= 3
