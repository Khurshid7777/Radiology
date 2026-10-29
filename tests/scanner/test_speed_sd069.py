"""SD-069: a board scan in parts runs faster with the same reading — the zone is cropped
to the page, header OCR stops once speed and gain are read, column runs are vectorised."""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.calibration import header_ocr
from ecgscan.layout import get_layout
from ecgscan.pipeline import _page_crop
from ecgscan.vectorize.trace import column_runs, runs_by_column

CROP = {"crop_pad_mm": 8.0}


def _zone_with_page(synth_mv, at=(400, 300)):
    """An A3 zone at 10 px/mm: white board, its grey outline along the edge, a 3x4 page on it."""
    zone = np.full((2170, 3400, 3), 250, np.uint8)
    zone[:3], zone[-3:], zone[:, :3], zone[:, -3:] = 150, 150, 150, 150
    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
    y, x = at
    ph, pw = min(page.shape[0], 2170 - y - 100), min(page.shape[1], 3400 - x - 100)
    zone[y:y + ph, x:x + pw] = page[:ph, :pw]
    return zone, (y, x, ph, pw)


def test_the_zone_is_cropped_to_the_page_with_its_margin(synth_mv):
    zone, (_, _, ph, pw) = _zone_with_page(synth_mv, at=(400, 300))
    out = _page_crop(zone, 10.0, CROP)
    assert out.shape[0] < zone.shape[0] and out.shape[1] < zone.shape[1]
    # The whole page is inside, with no more than the margin (+ the 4-px sampling step) around it.
    assert ph <= out.shape[0] <= ph + 2 * 80 + 8 and pw <= out.shape[1] <= pw + 2 * 80 + 8
    assert out.flags["C_CONTIGUOUS"]


def test_a_full_zone_an_empty_one_or_no_pad_is_left_as_it_is(synth_mv):
    zone, _ = _zone_with_page(synth_mv, at=(400, 300))
    assert _page_crop(zone, 10.0, {}) is zone                                       # crop_pad_mm unset
    empty = np.full((2170, 3400, 3), 250, np.uint8)
    empty[:3] = 150                                                                 # only the outline
    assert _page_crop(empty, 10.0, CROP) is empty
    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
    assert _page_crop(page, 10.0, CROP) is page                                     # the page is the zone


def test_column_runs_at_once_equal_column_by_column():
    rng = np.random.default_rng(0)
    for _ in range(40):
        sub = rng.random((int(rng.integers(1, 50)), int(rng.integers(1, 70)))) > rng.random()
        y0 = int(rng.integers(0, 90))
        assert runs_by_column(sub, y0) == [[(a + y0, b + y0) for a, b in column_runs(sub[:, i])]
                                           for i in range(sub.shape[1])]
    assert runs_by_column(np.zeros((5, 0), bool)) == []


class _FakeOCR:
    def __init__(self, tiles):
        self.tiles, self.calls = tiles, 0

    def __call__(self, img):
        text = self.tiles[min(self.calls, len(self.tiles) - 1)]
        self.calls += 1
        return [[None, t, 0.99] for t in text], None


@pytest.mark.parametrize(("tiles", "calls"), [
    ([["25mm/s", "10mm/mV"], ["Name"], ["ID"], ["x"]], 1),      # both in the first tile: stop there
    ([["25mm/s"], ["10mm/mV"], ["ID"], ["x"]], 2),              # gain in the second
])
def test_header_ocr_stops_once_speed_and_gain_are_read(monkeypatch, tiles, calls):
    fake = _FakeOCR(tiles)
    monkeypatch.setattr(header_ocr, "_engine", lambda *a, **k: fake)
    band = np.full((1000, 4000, 3), 255, np.uint8)
    r = header_ocr.read_bands(band, header_frac=0.3, footer_frac=0.1)
    assert fake.calls == calls and r.speeds == [25.0] and r.gains == [10.0]


def test_header_ocr_reads_every_tile_when_a_value_is_missing(monkeypatch):
    fake = _FakeOCR([["25mm/s"], ["Name"]])
    monkeypatch.setattr(header_ocr, "_engine", lambda *a, **k: fake)
    r = header_ocr.read_bands(np.full((1000, 4000, 3), 255, np.uint8), header_frac=0.3, footer_frac=0.1)
    # header 300 px: tiles 900 wide every 675 px over 4000 -> 6; footer 100 px: 300 every 225 -> 18
    assert fake.calls == 24 and r.speeds == [25.0] and not r.gains
