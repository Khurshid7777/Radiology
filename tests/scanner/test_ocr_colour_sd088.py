"""SD-088: speed and gain printed in black over a coloured grid are read again on the
brightest channel when the colour read leaves them out."""

from unittest import mock

import numpy as np

from ecgscan.calibration import header_ocr
from ecgscan.calibration.header_ocr import _coloured, read_bands


def _page(colour: bool) -> np.ndarray:
    page = np.full((720, 1400, 3), 255, np.uint8)
    if colour:
        page[600:, ::50] = (40, 40, 170)                  # red 5-mm lines in the footer band
    return page


class _Ocr:
    """Kit 12390: on the colour band only the digits come back; on the brightest channel the
    units do too. Records which images it was given."""

    def __init__(self):
        self.calls: list[bool] = []

    def __call__(self, img):
        red = bool(_coloured(img, 0.005))
        self.calls.append(red)
        return ([[None, "25", 0.9], [None, "10", 0.9]] if red else [[None, "25mm/s", 0.9], [None, "10mm/mV", 0.9]]), None


def _read(page, ocr, colour_frac):
    with mock.patch.object(header_ocr, "_engine", return_value=ocr):
        return read_bands(page, header_frac=0.0, footer_frac=120 / 720, min_band_px=40, min_tile_px=1400,
                          colour_frac=colour_frac)


def test_a_coloured_band_without_units_is_read_again_on_the_brightest_channel():
    ocr = _Ocr()
    r = _read(_page(colour=True), ocr, 0.02)
    assert ocr.calls == [True, False]                     # colour first, then red gone white
    assert r.speeds == [25.0] and r.gains == [10.0]


def test_bare_numbers_alone_are_not_settings():
    ocr = _Ocr()
    r = _read(_page(colour=True), ocr, 0.0)               # retry off
    assert r.speeds == [] and r.gains == []


def test_a_grey_band_is_not_read_twice():
    ocr = mock.MagicMock(return_value=([[None, "Name:", 0.9]], None))
    _read(_page(colour=False), ocr, 0.02)
    assert ocr.call_count == 1
