"""Grid pitch against the calibration pulses (D128): a blurred 4-px/mm strip reads its grid as
6.88 px/mm, the record runs 1.72x fast and Einthoven still holds; the pulses then imply a gain
no recorder prints, and the photo is flagged instead of reporting a false tachycardia."""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import degrade_image
from cardiolens.layer1_ingestion.strip import (
    SCALE_CONFLICT_NOTE,
    digitize_strip,
    digitize_strip_to_signal,
    render_ecg_strip,
)
from cardiolens.pipeline import run_pipeline
from cardiolens.telegram.analyze import format_card
from cardiolens.telegram.store import Record


@pytest.fixture(scope="module")
def sig():
    return generate_synthetic_ecg(heart_rate_bpm=60, duration_s=10.0, seed=1)


def _card(result):
    rec = Record(id="1_2", chat_id=1, msg_id=2, chat_title="ЭКГ чат", date="2026-10-20T10:00:00",
                 predicted=sorted(result.findings.codes()))
    return format_card(rec, result)


def test_blurred_small_strip_is_flagged_not_read_fast(sig, tmp_path):
    clean = render_ecg_strip(sig, tmp_path / "s.png", px_per_mm=4)
    blurred = degrade_image(clean, tmp_path / "b.png", blur_px=3)
    s = digitize_strip_to_signal(blurred)
    assert digitize_strip.last_calibration.get("scale_conflict")
    assert SCALE_CONFLICT_NOTE in s.provenance.notes
    r = run_pipeline(s, mode="A")
    assert r.findings.codes() == {"photo_unreliable"} and not r.safety.alerts
    assert "масштаб не определён" in r.findings.findings[0].label_ru
    card = _card(r)
    assert "ЧСС и интервалы не измерены" in card and "103" not in card


def test_sharp_strip_has_no_conflict(sig, tmp_path):
    s = digitize_strip_to_signal(render_ecg_strip(sig, tmp_path / "s.png", px_per_mm=4))
    assert not digitize_strip.last_calibration.get("scale_conflict")
    r = run_pipeline(s, mode="A")
    assert abs(r.measurements.heart_rate_bpm - 60) <= 2
    assert "photo_unreliable" not in r.findings.codes()
    assert "(по ненадёжному снимку)" not in _card(r)
