"""Report sheets (D126): twelve rows in a block at the left, text on the right, a 10-s rhythm
strip at 12.5 mm/s under the text (Cardipia 800H); the photo is often turned by a quarter."""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.sheet import (
    digitize_sheet,
    digitize_sheet_to_signal,
    render_report_sheet,
)
from cardiolens.layer1_ingestion.strip import (
    digitize_auto_to_signal,
    digitize_strip,
    render_ecg_strip,
)
from cardiolens.pipeline import run_pipeline


@pytest.mark.parametrize("turn", [0, 1, 3])
def test_sheet_is_found_at_any_quarter_turn(tmp_path, turn):
    sig = generate_synthetic_ecg(heart_rate_bpm=96, duration_s=10.0, seed=3)
    path = render_report_sheet(sig, tmp_path / "s.png", turn=turn)
    digitize_sheet(path)
    c = digitize_sheet.last_calibration
    assert (c["rotated_deg"] + 90 * turn) % 360 == 0
    s = digitize_sheet_to_signal(path)
    assert s.provenance.lead_windows["II"][1] > 9.5        # II continued by the rhythm strip
    assert s.provenance.lead_windows["V1"][1] < 5.0
    assert len(s.provenance.discontinuities_s) == 1


@pytest.mark.parametrize("hr,seed,code", [(60, 5, "sinus_rhythm"), (45, 9, "sinus_bradycardia"),
                                          (130, 7, "sinus_tachycardia")])
def test_sheet_runs_through_the_pipeline(tmp_path, hr, seed, code):
    sig = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10.0, seed=seed)
    s = digitize_auto_to_signal(render_report_sheet(sig, tmp_path / "s.png", turn=1))
    assert any(n.startswith("report sheet photo") for n in s.provenance.notes)
    r = run_pipeline(s, mode="A")
    assert abs(r.measurements.heart_rate_bpm - hr) <= 3
    assert code in r.findings.codes() and "photo_unreliable" not in r.findings.codes()


def test_plain_strip_is_not_taken_for_a_sheet(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0, seed=1)
    path = render_ecg_strip(sig, tmp_path / "strip.png")
    with pytest.raises(ValueError):
        digitize_sheet(path)
    s = digitize_auto_to_signal(path)
    assert any("strip: all leads simultaneous" in n for n in s.provenance.notes)
    assert digitize_strip.last_calibration["layout"] == "12x1"


@pytest.mark.parametrize("fs,low", [(110.0, True), (200.0, False)])
def test_unreliable_small_photo_asks_for_a_closer_shot_sent_as_a_file(fs, low):
    import numpy as np

    from cardiolens.contracts.signal import Provenance

    sig = generate_synthetic_ecg(heart_rate_bpm=80, duration_s=10.0, seed=2, sampling_rate_hz=fs)
    x = np.array(sig.signal, dtype=np.float32)
    x[sig.lead_names.index("III")] = np.random.default_rng(0).normal(0, 300, x.shape[1])   # breaks Einthoven
    sig = sig.model_copy(update={"signal": x, "source": "image", "provenance": Provenance(
        digitized=True, lead_windows={ln: (0.0, 10.0) for ln in sig.lead_names})})
    r = run_pipeline(sig, mode="A")
    (f,) = r.findings.findings
    assert f.code == "photo_unreliable"
    assert ("отправьте как файл" in f.label_ru) is low
