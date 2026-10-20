"""FastAPI service tests (TZ §14)."""

import os

import pytest
from fastapi.testclient import TestClient

from cardiolens.api.app import app

client = TestClient(app, headers={"X-API-Key": "test-key"})
_SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_cases_listed():
    r = client.get("/api/cases")
    assert r.status_code == 200
    assert {"normal", "hyperk", "stemi"} <= set(r.json()["cases"])


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "CardioLens" in r.text


def test_analyze_synthetic_hyperk_mode_b_escalates():
    r = client.post("/api/analyze", data={"case": "hyperk", "mode": "B"})
    assert r.status_code == 200
    d = r.json()
    assert d["mode"] == "escalate"
    assert d["escalation_reason"]
    assert any(a["code"] == "peaked_t_waves" for a in d["critical_alerts"])
    assert d["conclusion_ru"] and d["conclusion_uz"]
    assert "analysis_id" in d


def test_analyze_stemi_critical():
    r = client.post("/api/analyze", data={"case": "stemi", "mode": "A"})
    d = r.json()
    assert d["mode"] == "A"
    assert any(a["code"] == "st_elevation" for a in d["critical_alerts"])


def test_analyze_requires_input():
    r = client.post("/api/analyze", data={"mode": "A"})
    assert r.status_code == 400


def test_analyze_csv_upload_and_artifacts():
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post(
            "/api/analyze",
            data={"mode": "A", "sampling_rate_hz": "500", "unit": "uv"},
            files={"files": ("sample_ecg.csv", f, "text/csv")},
        )
    assert r.status_code == 200
    d = r.json()
    assert d["measurements"]["heart_rate_bpm"] == pytest.approx(75, abs=4)
    aid = d["analysis_id"]
    # All downstream artifacts are retrievable.
    assert client.get(f"/api/analysis/{aid}/report.pdf").status_code == 200
    assert client.get(f"/api/analysis/{aid}/waveform.svg").status_code == 200
    assert client.get(f"/api/analysis/{aid}/graph.png").status_code == 200
    assert client.get(f"/api/analysis/{aid}/conclusion?lang=uz").status_code == 200


def test_csv_upload_without_rate_rejected():
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A"},
                        files={"files": ("x.csv", f, "text/csv")})
    assert r.status_code == 400


def test_missing_analysis_404():
    assert client.get("/api/analysis/deadbeef/report.pdf").status_code == 404


def test_feedback_and_calibration_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIOLENS_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    r = client.post("/api/analyze", data={"case": "stemi", "mode": "A"})
    aid = r.json()["analysis_id"]
    fb = {
        "analysis_id": aid,
        "reviewer": "dr. Y",
        "findings": [
            {"code": "st_elevation", "verdict": "confirmed", "confidence": 0.8},
            {"code": "short_qt", "verdict": "rejected", "confidence": 0.7},
        ],
        "overall_agreement": True,
    }
    rf = client.post(f"/api/analysis/{aid}/feedback", json=fb)
    assert rf.status_code == 200 and rf.json()["status"] == "recorded"
    # Calibration endpoint runs (too few samples -> stays identity).
    rc = client.post("/api/calibration/fit")
    assert rc.status_code == 200
    assert "temperature" in rc.json() and "ece_before" in rc.json()
