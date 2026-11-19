"""D322 (after D321's review): beats near the strip's start and end that the R detector misses are added from a second,
mirror-padded run; beats of the plain run are never removed, so a pause ending near the end keeps its alert."""
import importlib

import numpy as np

from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure

M = importlib.import_module("cardiolens.layer3_measure.measure")


def test_plain_run_beats_are_never_removed(monkeypatch):
    for seed in range(5):
        sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=seed)
        with_edge = set(measure(sig).r_peaks_samples)
        monkeypatch.setattr(M, "R_EDGE_PAD_S", 0.0)
        plain = set(measure(sig).r_peaks_samples)
        monkeypatch.setattr(M, "R_EDGE_PAD_S", 1.0)
        assert plain <= with_edge and len(with_edge - plain) <= 2


def test_edge_beats_are_added_inside_the_edge_zones_only():
    calls = []

    class FakeNk:
        @staticmethod
        def ecg_peaks(x, sampling_rate):
            calls.append(len(x))
            pad = int(M.R_EDGE_PAD_S * sampling_rate)
            # the padded run "finds" beats at 0.1 s (too close), 0.4 s (ok), 5 s (inside, ignored), 9.9 s (too close)
            return None, {"ECG_R_Peaks": np.array([int(t * sampling_rate) + pad for t in (0.1, 0.4, 5.0, 9.9)])}
    fs = 500.0
    x = np.zeros(int(10 * fs))
    out = M._edge_beats(x, np.array([1000, 2000]), fs, FakeNk)
    assert list(out) == [200, 1000, 2000] and calls == [len(x) + 2 * int(fs)]
    assert list(M._edge_beats(x[:int(3 * fs)], np.array([500]), fs, FakeNk)) == [500]     # short record: untouched


def test_mirror_maps_back_exactly():
    fs, pad, n = 500.0, 500, 5001
    x = np.arange(n, dtype=float)
    padded = np.concatenate([x[pad:0:-1], x, x[-2:-pad - 2:-1]])
    assert len(padded) == n + 2 * pad and (padded[pad:pad + n] == x).all()


def test_a_pause_ending_near_the_end_keeps_its_alert(monkeypatch):
    """The review's case: a 3-s pause whose closing beat lies 0.08-0.19 s before the end; the plain run finds it."""
    from cardiolens.pipeline import run_pipeline
    base = generate_synthetic_ecg(heart_rate_bpm=60, duration_s=20, seed=3)
    fs = base.sampling_rate_hz
    monkeypatch.setattr(M, "R_EDGE_PAD_S", 0.0)
    tr = np.asarray(measure(base).r_peaks_samples)
    monkeypatch.setattr(M, "R_EDGE_PAD_S", 1.0)
    x, k = base.signal, 10
    cut = int((tr[k] + tr[k + 1]) // 2)
    y = np.concatenate([x[:, :cut], np.repeat(x[:, cut:cut + 1], int(3.0 * fs), axis=1), x[:, cut:]], axis=1)
    for e in (0.08, 0.16):
        end = int(tr[k + 1] + 3.0 * fs + e * fs)
        s = base.model_copy(update={"signal": y[:, end - int(10 * fs):end].copy(), "duration_s": 10.0})
        res = run_pipeline(s, mode="A")
        assert "sinus_pause" in res.findings.codes() or "ventricular_pause" in {al.code for al in res.safety.alerts}, e
