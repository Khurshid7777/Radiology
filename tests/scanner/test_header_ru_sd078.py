"""SD-078: speed and gain printed in Russian ("50.0 мм/сек", "20.0 мм/мВ") are read from the header.

The Latin-script OCR model returns the Cyrillic units as look-alikes ("MM/ceK", "MM/c", "MM/MB");
before SD-078 neither was matched and a 50 mm/s strip read without hints at 25 (time x2)."""

from pathlib import Path

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.calibration.header_ocr import ocr_available, parse_settings
from ecgscan.layout import get_layout
from ecgscan.pipeline import scan

cv2 = pytest.importorskip("cv2")
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@pytest.mark.parametrize(("texts", "speeds", "gains"), [
    (["20.0MM/MB, 50.0 MM/cek"], [50.0], [20.0]),           # RapidOCR on "20.0 мм/мВ, 50.0 мм/сек"
    (["20.0 MM/MB, 50.0 MM/ceK"], [50.0], [20.0]),
    (["CKopocTb: 25 MM/c ycmneHe: 10 MM/MB"], [25.0], [10.0]),
    (["50MM/c20MM/MB"], [50.0], [20.0]),
    (["20,0 мм/мВ 50,0 мм/сек"], [50.0], [20.0]),           # an engine that returns Cyrillic
    (["25 мм/с 10 мм/мв"], [25.0], [10.0]),
    (["25 mm/sec 10mm/mV"], [25.0], [10.0]),
])
def test_russian_units_and_their_ocr_look_alikes(texts, speeds, gains):
    assert parse_settings(texts) == (speeds, gains)


def test_no_unit_that_only_starts_alike():
    assert parse_settings(["10 mm/cm", "5 mm/mbar", "50 MM/ 20 MM/"]) == ([], [])


@pytest.mark.skipif(not FONT.exists() or not ocr_available()[0], reason="needs DejaVu Sans and RapidOCR")
def test_a_russian_header_sets_speed_and_gain(synth_mv):
    """A 50 mm/s, 20 mm/mV page whose header says so in Russian only."""
    from PIL import Image, ImageDraw, ImageFont

    page, gt = render_page(synth_mv, 500.0, get_layout("6x2"),
                           RenderSpec(px_per_mm=10.0, speed_mm_s=50, gain_mm_mv=20, header_settings=False))
    x0, y0, _, y1 = gt.header_box_px
    img = Image.fromarray(page[:, :, ::-1].copy())
    ImageDraw.Draw(img).text((x0, y0 + (y1 - y0) // 2), "Скорость 50.0 мм/сек   Усиление 20.0 мм/мВ", fill=(20, 20, 20),
                             font=ImageFont.truetype(str(FONT), int(3.2 * 10)))
    res = scan(cv2.imencode(".png", np.array(img)[:, :, ::-1])[1].tobytes())
    cal = res.calibration
    assert cal.paper_speed_mm_s == 50 and cal.speed_source == "header_ocr"
    assert cal.gain_mm_mv == 20
