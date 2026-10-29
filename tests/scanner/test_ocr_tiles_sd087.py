"""SD-087: OCR tiles overlap enough that a printed setting is whole in some tile (Kit 12390 at
10 px/mm: "25mm/s" at x 500-650 and "10mm/mV" at x 1000-1185 lay across every edge of the
old 360-px tiles, which overlapped by 90 px)."""

from ecgscan.calibration.header_ocr import tiles


def _whole(span: tuple[int, int], ts: list[tuple[int, int]]) -> bool:
    return any(x0 <= span[0] and span[1] <= x1 for x0, x1 in ts)


def test_the_old_tiles_cut_both_settings():
    ts = tiles(2794, 120)
    assert not _whole((500, 650), ts) and not _whole((1000, 1185), ts)


def test_any_setting_as_wide_as_the_overlap_is_whole_in_some_tile():
    ts = tiles(2794, 120, min_tile_px=600, overlap_px=300)
    assert ts[0][0] == 0 and ts[-1][1] == 2794
    assert all(_whole((x, x + 300), ts) for x in range(2794 - 300))


def test_a_tall_band_keeps_its_wide_tiles():
    assert tiles(2794, 540, min_tile_px=600, overlap_px=300) == [(0, 1620), (1215, 2794)]
