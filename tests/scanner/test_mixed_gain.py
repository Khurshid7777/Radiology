"""SD-153: mixed gain - the chest leads printed at half the limb gain ("10/5 mm/mV")."""

import pytest

from ecgscan.calibration.header_ocr import mixed_gain, parse_settings
from ecgscan.contract.models import WARNING_CODES, Calibration
from ecgscan.i18n import strings
from ecgscan.pipeline import chest_gain, rescale_chest

GAINS = [5, 10, 20]


@pytest.mark.parametrize("text, pair", [
    ("25 mm/s   10/5 mm/mV   0.05-150 Hz", (10.0, 5.0)), ("10/5мм/мВ", (10.0, 5.0)), ("20/10 mm/mV", (20.0, 10.0)),
    ("10 / 5 мм/мВ", (10.0, 5.0)), ("1O/5 mm/mV", (10.0, 5.0)),
    ("0.05/150 Hz", None), ("25 mm/s 10 mm/mV", None), ("5/10 mm/mV", None), ("10/5 mm/s", None), ("10/4 mm/mV", None),
])
def test_the_header_names_a_mixed_gain_only_as_a_and_its_half(text, pair):
    assert mixed_gain([text], GAINS) == pair


def test_a_mixed_header_alone_would_have_said_the_chest_gain():
    assert parse_settings(["10/5 mm/mV"])[1] == [5.0]          # why the pipeline takes the pair's limb gain instead


def test_whose_word_counts():
    assert chest_gain((10.0, 5.0), None, 10.0) == 5.0
    assert chest_gain((10.0, 5.0), None, 20.0) is None              # the header's limb gain is not the sheet's
    assert chest_gain(None, 5.0, 10.0) == 5.0 and chest_gain((10.0, 5.0), 10.0, 10.0) is None
    assert chest_gain(None, None, 10.0) is None and chest_gain((10.0, 5.0), None, None) is None


def test_only_the_chest_leads_are_rescaled():
    from types import SimpleNamespace as N

    rs = [N(panel=N(lead="V2"), trace=N(samples_mv=[0.5, None, -0.25])),
          N(panel=N(lead="II"), trace=N(samples_mv=[0.5, None]))]
    rescale_chest(rs, 2.0)
    assert rs[0].trace.samples_mv == [1.0, None, -0.5] and rs[1].trace.samples_mv == [0.5, None]


def test_the_contract_and_the_words():
    assert {"MIXED_GAIN", "NARROW_FILTER"} <= set(WARNING_CODES)
    assert Calibration(gain_mm_mv=10, gain_chest_mm_mv=5, confidence=0.9).gain_chest_mm_mv == 5
    for lang in ("ru", "uz"):
        assert "V1" in strings(lang)["warning.MIXED_GAIN"]


@pytest.mark.parametrize("layout", ["3x4+rhythm", "6x2"])
def test_a_mixed_sheet_is_read_at_its_own_scales(layout):
    pytest.importorskip("cv2")
    import importlib.util
    from pathlib import Path

    from ecgscan.pipeline import ScanHints, scan

    spec = importlib.util.spec_from_file_location("mg", Path(__file__).parents[2] / "tools" / "scanner" / "mixed_gain_check.py")
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    data, gt = mg.sheet(901, layout, chest=5.0, header_pair=True)
    res = scan(data, hints=ScanHints(paper_speed_mm_s=25))
    err = mg.r_peak_error(gt, res)
    assert res.calibration.gain_mm_mv == 10 and res.calibration.gain_chest_mm_mv == 5.0 and "MIXED_GAIN" in res.warnings
    assert err["chest"] < 0.2 and err["limb"] < 0.2
    data, gt = mg.sheet(901, layout, chest=5.0, header_pair=False)                # nothing says the gain is mixed
    res = scan(data, hints=ScanHints(paper_speed_mm_s=25))
    assert res.calibration.gain_chest_mm_mv is None and mg.r_peak_error(gt, res)["chest"] > 0.4


def test_a_doctors_chest_gain_the_contract_does_not_know_is_ignored():
    assert chest_gain(None, 20.0, 10.0) is None and chest_gain(None, 7.0, 10.0) is None    # review of D353
    assert chest_gain(None, 10.0, 20.0) == 10.0 and chest_gain(None, 2.5, 5.0) == 2.5
