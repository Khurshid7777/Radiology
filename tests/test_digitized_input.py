"""A signal digitised from a photo (ECG scanner M4, D130, scanner SD-015).

* never autonomous: mode B escalates, the report confidence is capped by the
  digitisation's, and every report carries the RU/UZ "from a photo" mark;
* on a page, a single-lead beat at the edge of the rhythm lead's own window counts
  only when the page columns confirm it, and the final beats are checked against
  the columns.
"""

import json
from pathlib import Path

import numpy as np

from cardiolens.config import load_app_config
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer2_preprocess import preprocess
from cardiolens.layer3_measure import measure
from cardiolens.outputs.conclusion import DIGITIZED_NOTE, format_conclusion
from cardiolens.outputs.json_out import result_to_dict
from cardiolens.pipeline import run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "digitized_6x2_page.npz"


def _digitized(confidence: float = 0.6) -> SignalRepresentation:
    sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=3)
    sig.provenance.digitized = True
    sig.provenance.confidence = confidence
    return sig


def test_photo_input_never_runs_autonomously():
    res = run_pipeline(_digitized(), mode="B")
    assert res.report.mode == "escalate"
    assert "фото" in res.report.escalation_reason


def test_photo_input_confidence_is_capped_by_the_digitisation():
    res = run_pipeline(_digitized(0.4), mode="A")
    assert res.causal_graph.confidence <= 0.4


def test_photo_input_is_marked_in_both_languages_and_json():
    res = run_pipeline(_digitized(), mode="A")
    assert DIGITIZED_NOTE[0] in format_conclusion(res, "ru")
    assert DIGITIZED_NOTE[1] in format_conclusion(res, "uz")
    d = result_to_dict(res)
    assert d["digitized_from_image"] is True and d["digitization_confidence"] == 0.6


def test_digital_input_is_unmarked():
    res = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72, seed=3), mode="A")
    assert DIGITIZED_NOTE[0] not in format_conclusion(res, "ru")
    assert result_to_dict(res)["digitized_from_image"] is False


def _fixture_page() -> SignalRepresentation:
    z = np.load(FIXTURE)
    sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=3)       # carrier for the metadata
    sig.signal = z["signal"].astype(np.float32)
    sig.lead_names = [str(x) for x in z["lead_names"]]
    sig.provenance.digitized = True
    sig.provenance.lead_windows = {k: tuple(v) for k, v in json.loads(str(z["windows"])).items()}
    cfg = load_app_config()
    pp = cfg.preprocessing
    return preprocess(sig, mains_frequency_hz=cfg.mains_frequency_hz, bandpass_low_hz=pp.bandpass_low_hz,
                      bandpass_high_hz=pp.bandpass_high_hz, target_sampling_rate_hz=pp.target_sampling_rate_hz)


def test_beat_at_the_end_of_the_rhythm_leads_window_needs_the_columns():
    """A scanned 6x2 page (lead II only 0-5 s): the step into the padding at 5 s was
    a beat, it blocked the columns' beats and the page read HR 77 for 68.6."""
    m = measure(_fixture_page())
    assert abs(m.heart_rate_bpm - 68.6) < 1.0
    assert m.page_beat_agreement is True
    assert not any(2450 <= r <= 2550 for r in m.r_peaks_samples)


def test_beats_disagreeing_with_the_columns_are_flagged():
    sig = _fixture_page()
    page_windows = sig.provenance.lead_windows
    from cardiolens.layer3_measure.measure import _page_beat_agreement, _pick_rhythm_lead

    row = sig.signal[_pick_rhythm_lead(sig)].astype(np.float64)
    good = np.array(measure(sig).r_peaks_samples)
    assert _page_beat_agreement(sig, page_windows, row, good, 500.0) is True
    extra = np.sort(np.append(good, (good[1] + good[2]) // 2))      # a beat inside a normal RR
    assert _page_beat_agreement(sig, page_windows, row, extra, 500.0) is False
