"""D198: the quality summary counts a number shown to the user that disagrees with the recorder's print."""

from fastapi.testclient import TestClient

from cardiolens.api.app import app
from cardiolens.feedback import quality


def _row(gates=(), mismatch=("hr_bpm",), released=False):
    return {"ts": 10, "route": "scanner", "gates": list(gates), "hr_bpm": 90, "qrs_ms": 100,
            "device": {"hr_bpm": 70, "qrs_ms": 98}, "device_mismatch": list(mismatch), "rate_released": released}


def test_a_shown_mismatch_is_counted_and_a_gated_one_is_not():
    s = quality.summarize([_row(), _row(gates=["photo_unreliable"]), _row(mismatch=())])
    hr = s["device"]["hr_bpm"]
    assert hr["n"] == 3 and hr["shown_mismatch"] == 1 and hr["agree"] == 1


def test_a_released_rate_counts_as_shown():
    s = quality.summarize([_row(gates=["photo_unreliable"], released=True)])
    assert s["device"]["hr_bpm"]["shown_mismatch"] == 1
    assert s["device"]["qrs_ms"]["shown_mismatch"] == 0        # intervals stay hidden


def test_a_non_hiding_flag_still_counts_as_shown():
    assert quality.summarize([_row(gates=["screen_photo_limited"])])["device"]["hr_bpm"]["shown_mismatch"] == 1


def test_the_summary_needs_a_key(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "tester:test-key")
    assert TestClient(app).get("/api/quality/summary").status_code == 401
