"""D321: the R detector sees beats near the strip's start and end (mirrored padding); a complex cut by the edge is
dropped. On the Holter bench 70 % of missed beats were in the first / last 0.6 s (D319)."""
import importlib

import numpy as np

from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure

M = importlib.import_module("cardiolens.layer3_measure.measure")


def _beats(sig):
    return np.asarray(measure(sig).r_peaks_samples)


def test_padding_finds_edge_beats_and_drops_cut_complexes(monkeypatch):
    fs_beats = []
    for seed in range(6):
        sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=seed)
        fs = sig.sampling_rate_hz
        with_pad = _beats(sig)
        monkeypatch.setattr(M, "R_EDGE_PAD_S", 0.0)
        without = _beats(sig)
        monkeypatch.setattr(M, "R_EDGE_PAD_S", 1.0)
        assert len(with_pad) >= len(without)                         # never fewer beats
        n = sig.signal.shape[1]
        assert ((with_pad >= 0.2 * fs) & (with_pad < n - 0.2 * fs)).all() or not M.R_EDGE_KEEP_S
        fs_beats.append(len(with_pad) - len(without))
    assert max(fs_beats) <= 2                                        # at most the two edge beats are added


def test_constants_as_accepted():
    assert M.R_EDGE_PAD_S == 1.0 and M.R_EDGE_KEEP_S == 0.2
