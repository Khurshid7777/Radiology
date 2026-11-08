"""D210 UI P3: the norms in the answer are the rules' own ranges; the waveform samples are the SVG's
leads, and none under a photo gate."""

import importlib

import pytest
from fastapi.testclient import TestClient

from cardiolens.api.service import signal_from_case
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import SCALE_CONFLICT_NOTE
from cardiolens.outputs.display import display_norms, waveform_samples
from cardiolens.pipeline import run_pipeline

api = importlib.import_module("cardiolens.api.app")


def test_adult_norms_are_the_tables():
    sig = signal_from_case("normal")
    sig.patient.age_years, sig.patient.sex = 54, "male"
    n = display_norms(run_pipeline(sig, mode="A"))
    assert n["heart_rate_bpm"] == [60, 100] and n["pr_ms"] == [120, 200] and n["qrs_ms"] == [70, 110]
    assert n["qtc_ms"] == [330, 450] and n["qrs_axis_deg"] == [-30, 90] and n["table"] == "adult"


def test_no_sex_no_qtc_scale():
    sig = signal_from_case("normal")
    sig.patient.sex = None
    assert display_norms(run_pipeline(sig, mode="A"))["qtc_ms"] is None


def test_samples_follow_the_svg_leads_at_250_hz():
    w = waveform_samples(run_pipeline(signal_from_case("stemi"), mode="A"))
    assert not w["gated"] and list(w["leads"])[:3] == ["II", "III", "aVF"] and w["hz"] == pytest.approx(250, rel=0.01)
    assert len(w["leads"]["II"]["uv"]) == pytest.approx(5 * 250, abs=2)


def test_a_gated_photo_gets_no_samples():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    sig.provenance.notes = [SCALE_CONFLICT_NOTE]
    w = waveform_samples(run_pipeline(sig, mode="A"))
    assert w["gated"] and w["leads"] == {}


def test_the_endpoint_and_the_answer(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "a:key-a,b:key-b")
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    d = a.post("/api/analyze", data={"case": "normal", "mode": "A", "age_years": "54", "sex": "female"}).json()
    assert d["norms"]["qtc_ms"] == [330, 460]
    r = a.get(f"/api/analysis/{d['analysis_id']}/waveform.json")
    assert r.status_code == 200 and r.json()["leads"]
    b = TestClient(api.app, headers={"X-API-Key": "key-b"})
    assert b.get(f"/api/analysis/{d['analysis_id']}/waveform.json").status_code == 404
