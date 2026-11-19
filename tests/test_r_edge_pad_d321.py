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
    pad, n = 500, 5001
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


def test_a_tall_t_wave_at_the_start_is_not_added(monkeypatch):
    """D322 review: a strip starting on a QRS of a slow rhythm with T as tall as R — the T at ~0.34 s is not a beat."""
    from cardiolens.pipeline import run_pipeline
    b = generate_synthetic_ecg(duration_s=24, seed=2, heart_rate_bpm=50, t_amp_uv=1000, amplitude_scale=0.6)
    fs, i = b.sampling_rate_hz, b.lead_names.index("II")
    a0 = int(round(3.80 * fs))
    s = b.model_copy(update={"signal": b.signal[i:i + 1, a0:a0 + int(7 * fs)].copy(), "duration_s": 7.0, "lead_names": ["II"]})
    on = run_pipeline(s, mode="A")
    monkeypatch.setattr(M, "R_EDGE_PAD_S", 0.0)
    off = run_pipeline(s, mode="A")
    assert on.measurements.r_peaks_samples == off.measurements.r_peaks_samples
    assert on.findings.codes() == off.findings.codes()
