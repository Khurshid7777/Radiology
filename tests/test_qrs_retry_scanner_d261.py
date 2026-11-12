"""D261: the D236 QRS retry is back for a clean read of the ECG scanner; the readers' pages and the
scanner's fallback / grid / coarse reads keep the pre-D236 width (D256)."""
import sys

import numpy as np

import cardiolens.layer3_measure.measure  # noqa: F401
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg

M = sys.modules["cardiolens.layer3_measure.measure"]


def _digitised(notes):
    sig = generate_synthetic_ecg(bbb="lbbb", qrs_width_scale=1.9)
    prov = sig.provenance.model_copy(update={"digitized": True, "notes": notes})
    return sig.model_copy(update={"provenance": prov})


SCAN = "ecgscan scan_id=abc status=ok pipeline=1"


def test_clean_scanner_read():
    assert M._clean_scanner_read(_digitised([SCAN, "layout: 3x4 conf=0.9"]))
    assert M._clean_scanner_read(_digitised([SCAN, "warnings: QRS_UNSTABLE, RHYTHM_STRIP_ABSENT"]))
    assert not M._clean_scanner_read(_digitised([SCAN, "warnings: LOW_RESOLUTION_MARGINAL, SEGMENTER_FALLBACK"]))
    assert not M._clean_scanner_read(_digitised([SCAN, "warnings: GRID_IN_TRACE"]))
    assert not M._clean_scanner_read(_digitised([SCAN, "warnings: LOW_RESOLUTION_COARSE"]))
    assert not M._clean_scanner_read(_digitised(["3x4 page: each lead valid only inside it"]))
    assert not M._clean_scanner_read(_digitised([]))


def _retried(monkeypatch, notes) -> bool:
    calls = []

    def fake(signal, rpeaks, fs, *, hump_fraction=None, search_ms=None, max_ms=None):
        calls.append(search_ms)
        rp = np.asarray(rpeaks, dtype=float)
        if search_ms is None:
            return np.full(len(rp), np.nan), np.full(len(rp), np.nan)
        return rp - 0.04 * fs, rp + 0.15 * fs

    monkeypatch.setattr(M, "qrs_bounds_envelope", fake)
    M.measure(_digitised(notes))
    return M.QRS_RETRY_SEARCH_MS in calls


def test_retry_on_clean_scanner_read_only(monkeypatch):
    assert _retried(monkeypatch, [SCAN])
    assert not _retried(monkeypatch, [SCAN, "warnings: SEGMENTER_FALLBACK"])
    assert not _retried(monkeypatch, ["3x4 page: each lead valid only inside it"])
