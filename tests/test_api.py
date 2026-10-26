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
    # D167: the owner sees whether the learnable layer's model is on this server.
    assert r.json()["ml"]["model"] in ("loaded", "absent") and "path" in r.json()["ml"]


def test_json_never_carries_nan():
    """D165: a 6x2 screen photo gave sokolow_lyon_uv = NaN and /api/analyze a 500."""
    from cardiolens.outputs.json_out import finite

    assert finite({"a": float("nan"), "b": [1.0, float("inf"), {"c": -float("inf")}], "d": 2}) == \
        {"a": None, "b": [1.0, None, {"c": None}], "d": 2}


def test_cases_listed():
    r = client.get("/api/cases")
    assert r.status_code == 200
    assert {"normal", "hyperk", "stemi"} <= set(r.json()["cases"])


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "CardioVision" in r.text


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


# --- photos on the web page (D155) and the analysis thread (D156) ---------------------------

def _page_png(tmp_path, px_per_mm=10):
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer1_ingestion.digitize import render_ecg_page

    png = tmp_path / "page.png"
    render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), png,
                    px_per_mm=px_per_mm, calibration_pulse=True)
    return png


def test_photo_upload_is_read_like_the_bot_does(tmp_path):
    with open(_page_png(tmp_path), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A"},
                        files={"files": ("page.png", f, "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["measurements"]["heart_rate_bpm"] == pytest.approx(72, abs=4)
    assert d["conclusion_ru"]
    # D163: what was read, drawn over the page — for the doctor's eye, the bot and the PDF.
    assert d["overlay_available"] is True
    ov = client.get(f"/api/analysis/{d['analysis_id']}/overlay.jpg")
    assert ov.status_code == 200 and ov.headers["content-type"] == "image/jpeg"
    assert ov.content[:3] == b"\xff\xd8\xff"
    pdf = client.get(f"/api/analysis/{d['analysis_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_a_synthetic_case_has_no_overlay_and_the_overlay_never_reaches_json():
    from cardiolens.contracts.signal import Provenance

    r = client.post("/api/analyze", data={"case": "normal", "mode": "A"})
    d = r.json()
    assert d["overlay_available"] is False
    assert client.get(f"/api/analysis/{d['analysis_id']}/overlay.jpg").status_code == 404
    assert "overlay_jpeg" not in Provenance(overlay_jpeg=b"\xff").model_dump()


def test_photo_without_an_ecg_is_refused_in_words(tmp_path):
    import cv2
    import numpy as np

    wall = tmp_path / "wall.jpg"
    cv2.imwrite(str(wall), np.full((900, 1200, 3), 230, np.uint8))
    with open(wall, "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A"},
                        files={"files": ("wall.jpg", f, "image/jpeg")})
    assert r.status_code == 400
    assert any("а" <= ch <= "я" for ch in r.json()["detail"].lower()), r.text


def test_photo_and_signal_file_cannot_be_mixed(tmp_path):
    png = _page_png(tmp_path)
    with open(png, "rb") as f, open(os.path.join(_SAMPLES, "sample_ecg.csv"), "rb") as g:
        r = client.post("/api/analyze", data={"mode": "A", "sampling_rate_hz": "500"},
                        files=[("files", ("page.png", f, "image/png")),
                               ("files", ("sample_ecg.csv", g, "text/csv"))])
    assert r.status_code == 400 and "mixed" in r.json()["detail"]


def test_health_answers_while_an_analysis_runs(monkeypatch):
    import threading
    import time

    from cardiolens.api import service as svc

    real = svc.run_and_store

    def slow(*a, **k):
        time.sleep(1.5)
        return real(*a, **k)

    monkeypatch.setattr(svc, "run_and_store", slow)
    done = {}
    t = threading.Thread(target=lambda: done.setdefault(
        "r", client.post("/api/analyze", data={"case": "normal", "mode": "A"})))
    t.start()
    time.sleep(0.4)
    t0 = time.monotonic()
    h = client.get("/health")
    dt = time.monotonic() - t0
    t.join()
    assert h.status_code == 200 and dt < 1.0, dt
    assert done["r"].status_code == 200


# --- HEIC and PDF photos (D158) ---------------------------------------------------------------

def test_heic_and_pdf_photos_are_read(tmp_path):
    import pillow_heif
    import pymupdf
    from PIL import Image

    png = _page_png(tmp_path)
    pillow_heif.register_heif_opener()
    heic = tmp_path / "page.heic"
    Image.open(png).save(heic, quality=92)
    pdf = tmp_path / "page.pdf"
    with pymupdf.open() as doc:
        w, h = Image.open(png).size
        page = doc.new_page(width=w * 72 / 254, height=h * 72 / 254)   # 10 px/mm -> true size
        page.insert_image(page.rect, filename=str(png))
        doc.save(pdf)
    for name, mime in (("page.heic", "image/heic"), ("page.pdf", "application/pdf")):
        with open(tmp_path / name, "rb") as f:
            r = client.post("/api/analyze", data={"mode": "A"}, files={"files": (name, f, mime)})
        assert r.status_code == 200, (name, r.text)
        assert r.json()["measurements"]["heart_rate_bpm"] == pytest.approx(72, abs=4), name


# --- a messenger-sized photo is refused with the web page's wording (D159) --------------------

def test_small_photo_refusal_speaks_of_the_gallery_not_telegram(tmp_path):
    with open(_page_png(tmp_path, px_per_mm=4), "rb") as f:
        r = client.post("/api/analyze", data={"mode": "A"},
                        files={"files": ("page.png", f, "image/png")})
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "файлом" not in detail and "оригинал" in detail, detail
