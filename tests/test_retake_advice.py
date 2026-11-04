"""D192: the retake advice follows the reason a photo was gated; a photo with no trace found is
not told to check the electrodes."""

import cardiolens.pipeline as pl
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import (
    GRID_IN_TRACE_NOTE,
    LEADS_INCONSISTENT_NOTE,
    SCALE_CONFLICT_NOTE,
)


def _photo(*notes, fs=500.0):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, sampling_rate_hz=fs)
    sig.provenance.digitized = True
    sig.provenance.notes = list(notes)
    return sig


def _gate(sig):
    return pl.run_pipeline(sig, mode="A").findings.by_code("photo_unreliable")


def _qrs(monkeypatch, qrs_ms):
    real = pl.measure

    def m(*a, **k):
        r = real(*a, **k)
        r.intervals.qrs_ms = qrs_ms
        return r
    monkeypatch.setattr(pl, "measure", m)


def _assert_tip(f, key):
    ru, uz = pl.RETAKE_TIPS[key]
    assert f is not None and f.label_ru.endswith(ru) and f.label_uz.endswith(uz)


def test_a_scale_conflict_asks_for_the_whole_sheet_with_the_pulse():
    _assert_tip(_gate(_photo(SCALE_CONFLICT_NOTE)), "scale")


def test_inconsistent_leads_ask_for_a_flat_sheet():
    _assert_tip(_gate(_photo("ecgscan scan_id=x status=needs_confirmation",
                             "warnings: LIMB_LEADS_INCONSISTENT,RHYTHM_INCONSISTENT", LEADS_INCONSISTENT_NOTE)), "flat")


def test_grid_in_the_trace_asks_for_even_light_or_the_file():
    _assert_tip(_gate(_photo(GRID_IN_TRACE_NOTE)), "grid")


def test_an_impossible_qrs_asks_for_a_sharp_shot(monkeypatch):
    _qrs(monkeypatch, 12.0)
    _assert_tip(_gate(_photo()), "sharp")


def test_the_tips_differ_by_reason():
    assert len({ru for ru, _ in pl.RETAKE_TIPS.values()}) == len(pl.RETAKE_TIPS)
    assert len({uz for _, uz in pl.RETAKE_TIPS.values()}) == len(pl.RETAKE_TIPS)


def test_a_small_photo_still_asks_for_a_closer_shot_as_a_file():
    f = _gate(_photo(SCALE_CONFLICT_NOTE, fs=125.0))
    assert f is not None and "снимите ближе" in f.label_ru and "отправьте как файл" in f.label_ru
    assert pl.RETAKE_TIPS["scale"][0] not in f.label_ru


def test_no_trace_on_a_photo_is_about_the_shot_not_the_electrodes(monkeypatch):
    monkeypatch.setattr(pl, "_hf_amplitude_uv", lambda sig: 0.0)
    f = pl.run_pipeline(_photo(), mode="A").findings.by_code("no_ecg_signal")
    assert f.label_ru == pl.NO_TRACE_ON_PHOTO_RU and f.label_uz == pl.NO_TRACE_ON_PHOTO_UZ
    assert "электрод" not in f.label_ru and "elektrod" not in f.label_uz


def test_no_signal_in_a_digital_record_still_points_at_the_electrodes(monkeypatch):
    monkeypatch.setattr(pl, "_hf_amplitude_uv", lambda sig: 0.0)
    f = pl.run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.by_code("no_ecg_signal")
    assert "электроды" in f.label_ru and "elektrodlarni" in f.label_uz
