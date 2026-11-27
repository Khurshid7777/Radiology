"""D311: the scanner's LEAD_INCOHERENT warning gets its own text on the card instead of "check scale and layout"."""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline
from cardiolens.telegram.analyze import LEAD_INCOHERENT_RU, LEAD_INCOHERENT_UZ, format_card
from cardiolens.telegram.store import Record


@pytest.fixture(scope="module")
def result():
    return run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, seed=3), mode="A")


def _card(result, warnings):
    notes = result.signal.provenance.notes
    result.signal.provenance.notes = [n for n in notes if not n.startswith(("ecgscan", "warnings:"))] + \
        ["ecgscan scan_id=x status=needs_confirmation", *([f"warnings: {warnings}"] if warnings else [])]
    return format_card(Record(id="1_1", chat_id=1, msg_id=1), result)


def test_lead_incoherent_alone_has_its_own_text(result):
    card = _card(result, "LEAD_INCOHERENT")
    assert LEAD_INCOHERENT_RU in card and "масштаб и раскладку" not in card


def test_other_warnings_keep_the_general_line(result):
    card = _card(result, "LEAD_INCOHERENT, GRID_IN_TRACE")
    assert LEAD_INCOHERENT_RU in card and "масштаб и раскладку по снимку (GRID_IN_TRACE)" in card
    card = _card(result, "")
    assert "масштаб и раскладку по снимку" in card and LEAD_INCOHERENT_RU not in card


def test_uzbek_text_is_latin_only():
    assert not any("а" <= ch <= "я" for ch in LEAD_INCOHERENT_UZ.lower())
