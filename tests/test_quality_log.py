"""D190: the real-use quality log holds derived numbers only and never breaks an analysis."""

import json
import os

from fastapi.testclient import TestClient

from cardiolens.api.app import app
from cardiolens.feedback import quality
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline

client = TestClient(app, headers={"X-API-Key": "test-key"})
_SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def test_a_line_carries_numbers_and_the_recorders_values_but_no_names():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.device_measurements = {"hr_bpm": 95, "qrs_ms": 90}
    row = quality.quality_row(run_pipeline(sig, mode="A"), "telegram")
    assert row["client"] == "telegram" and row["route"] == "file"
    assert row["device"] == {"hr_bpm": 95, "qrs_ms": 90} and "hr_bpm" in row["device_mismatch"]
    text = json.dumps(row, ensure_ascii=False)
    for forbidden in ("patient", "filename", "conclusion", "narrative", "name"):
        assert forbidden not in text


def test_a_refusal_is_kept_as_a_kind_not_as_its_text():
    assert quality.refusal_row("Слишком низкое разрешение (меньше 8 пикселей на мм сетки)")["refused"] == "low_resolution"
    assert quality.refusal_row("Иванов_ЭКГ.jpg: что-то не так")["refused"] == "other"


def test_the_summary_counts_routes_gates_and_recorder_agreement():
    rows = [{"ts": 10, "route": "scanner", "gates": [], "hr_bpm": 70, "device": {"hr_bpm": 72}, "device_mismatch": []},
            {"ts": 10, "route": "scanner", "gates": ["photo_unreliable"], "hr_bpm": 90, "device": {"hr_bpm": 60},
             "device_mismatch": ["hr_bpm"]},
            {"ts": 10, "refused": "portrait"}]
    s = quality.summarize(rows)
    assert s["routes"]["scanner"] == {"read": 2, "gated": 1}
    assert s["refused"] == {"portrait": 1}
    assert s["device"]["hr_bpm"] == {"n": 2, "agree": 1, "gated": 1, "shown_mismatch": 0, "median_abs_err": 30}


def test_an_upload_writes_one_line_and_the_summary_reads_it(monkeypatch, tmp_path):
    path = tmp_path / "quality.jsonl"
    monkeypatch.setenv("CARDIOLENS_QUALITY_PATH", str(path))
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A", "sampling_rate_hz": "500", "unit": "uv"},
                        files={"files": ("sample_ecg.csv", f, "text/csv")})
    assert r.status_code == 200
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and "sample_ecg" not in lines[0]
    assert client.get("/api/quality/summary").json()["routes"]["file"]["read"] == 1


def test_a_log_that_cannot_be_written_does_not_fail_the_analysis(monkeypatch, tmp_path):
    (tmp_path / "f").write_text("")
    monkeypatch.setenv("CARDIOLENS_QUALITY_PATH", str(tmp_path / "f" / "quality.jsonl"))
    with open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A", "sampling_rate_hz": "500", "unit": "uv"},
                        files={"files": ("sample_ecg.csv", f, "text/csv")})
    assert r.status_code == 200
