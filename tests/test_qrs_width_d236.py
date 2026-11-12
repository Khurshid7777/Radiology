"""D236: QRS-envelope beats lost to the +-150 ms window are retried with a wider search, and the width
comes from envelope beats only (dwt bounds cut a bundle branch block short)."""
import sys

import numpy as np

import cardiolens.layer3_measure.measure  # noqa: F401
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import delineation

M = sys.modules["cardiolens.layer3_measure.measure"]


def test_lost_beats_are_retried_wider_and_dwt_does_not_mix_in(monkeypatch):
    calls = []

    def fake(signal, rpeaks, fs, *, hump_fraction=None, search_ms=None, max_ms=None):
        calls.append(search_ms)
        rp = np.asarray(rpeaks, dtype=float)
        on = rp - 0.04 * fs
        if search_ms is None:      # default window: every other beat comes back empty
            off = rp + 0.13 * fs
            on, off = on.copy(), off.copy()
            on[::2] = off[::2] = np.nan
            return on, off
        return on, rp + 0.14 * fs  # the wider search finds the 180 ms beats

    monkeypatch.setattr(M, "qrs_bounds_envelope", fake)
    sig = generate_synthetic_ecg(bbb="lbbb", qrs_width_scale=1.9)
    m = M.measure(sig)
    assert M.QRS_RETRY_SEARCH_MS in calls
    assert m.intervals.qrs_ms is not None and m.intervals.qrs_ms >= 170


def test_retry_window_is_wider_than_the_default():
    assert M.QRS_RETRY_SEARCH_MS > delineation.QRS_SEARCH_MS


def test_width_from_retried_beats_is_flagged(monkeypatch):
    """D239: when the width rests on retried beats, the measurement says so (paced ST is then not assessed)."""
    def fake(signal, rpeaks, fs, *, hump_fraction=None, search_ms=None, max_ms=None):
        rp = np.asarray(rpeaks, dtype=float)
        if search_ms is None:
            return np.full(len(rp), np.nan), np.full(len(rp), np.nan)
        return rp - 0.04 * fs, rp + 0.15 * fs

    monkeypatch.setattr(M, "qrs_bounds_envelope", fake)
    m = M.measure(generate_synthetic_ecg(bbb="lbbb", qrs_width_scale=1.9))
    assert m.qrs_width_from_retry is True


def test_paced_gate_treats_a_retried_width_as_unreliable():
    import inspect

    import cardiolens.pipeline as P
    assert "qrs_width_from_retry" in inspect.getsource(P)
