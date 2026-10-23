"""D147: a P wave repeats from beat to beat even when the RR is irregular; AF does not."""

import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings.p_consistency import p_window_consistency


def _r_peaks(sig):
    from cardiolens.pipeline import run_pipeline

    return run_pipeline(sig, mode="A").measurements.r_peaks_samples


def test_an_irregular_rhythm_with_p_waves_repeats():
    sig = generate_synthetic_ecg(heart_rate_bpm=70, rr_jitter_ms=120, seed=3)      # sinus arrhythmia
    corr, keep = p_window_consistency(sig, _r_peaks(sig))
    assert corr > 0.8 and keep > 0.6


def test_fibrillation_does_not_repeat():
    sig = generate_synthetic_ecg(heart_rate_bpm=90, p_amp_scale=0.0, rr_jitter_ms=180, noise_uv=25, seed=3)
    corr, keep = p_window_consistency(sig, _r_peaks(sig))
    assert corr < 0.6 and keep < 0.5


def test_missing_lead_or_too_few_beats_is_none():
    sig = generate_synthetic_ecg(seed=1)
    assert p_window_consistency(sig, [1000, 1500]) is None
    assert p_window_consistency(sig, list(np.arange(400, 4800, 400)), lead="X") is None
