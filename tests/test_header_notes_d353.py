"""D353: the scanner's header warnings (MIXED_GAIN, NARROW_FILTER) are plain lines on the card at any status."""

from pathlib import Path

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline
from cardiolens.telegram.analyze import HEADER_NOTES_RU, LEAD_INCOHERENT_RU, format_card
from cardiolens.telegram.store import Record


@pytest.fixture(scope="module")
def result():
    return run_pipeline(generate_synthetic_ecg(heart_rate_bpm=70, seed=3), mode="A")


def _card(result, status, warnings):
    notes = result.signal.provenance.notes
    result.signal.provenance.notes = [n for n in notes if not n.startswith(("ecgscan", "warnings:"))] + \
        [f"ecgscan scan_id=x status={status}", *([f"warnings: {warnings}"] if warnings else [])]
    return format_card(Record(id="1_1", chat_id=1, msg_id=1), result)


@pytest.mark.parametrize("status", ["ok", "needs_confirmation"])
def test_header_lines_at_any_status_without_raw_codes(result, status):
    card = _card(result, status, "MIXED_GAIN, NARROW_FILTER")
    assert HEADER_NOTES_RU["MIXED_GAIN"] in card and HEADER_NOTES_RU["NARROW_FILTER"] in card
    assert "MIXED_GAIN" not in card and "NARROW_FILTER" not in card


def test_doubt_still_said_when_only_header_notes(result):
    card = _card(result, "needs_confirmation", "MIXED_GAIN")
    assert "масштаб и раскладку по снимку" in card and "(MIXED_GAIN)" not in card
    card = _card(result, "needs_confirmation", "MIXED_GAIN, GRID_IN_TRACE")
    assert "масштаб и раскладку по снимку (GRID_IN_TRACE)" in card


def test_without_header_codes_card_unchanged(result):
    card = _card(result, "ok", "")
    assert not any(t in card for t in HEADER_NOTES_RU.values())
    card = _card(result, "needs_confirmation", "LEAD_INCOHERENT")
    assert LEAD_INCOHERENT_RU in card and "масштаб и раскладку" not in card


def test_desk_agent_not_in_this_batch():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "ecgscan_agent").exists()
    assert "agent/upload" not in (root / "src" / "ecgscan_service" / "router.py").read_text()
