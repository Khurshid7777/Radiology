"""D196: the waveform draws and marks the leads the findings name. D197: a signed 7-day link
gives the owner's PDF without a key and without the photo."""

import importlib
import time

import pytest
from fastapi.testclient import TestClient

from cardiolens.api import service
from cardiolens.api.service import signal_from_case
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import SCALE_CONFLICT_NOTE
from cardiolens.outputs.waveform import finding_leads
from cardiolens.pipeline import run_pipeline

api = importlib.import_module("cardiolens.api.app")


def test_stemi_draws_and_marks_its_inferior_leads():
    leads, marked = finding_leads(run_pipeline(signal_from_case("stemi"), mode="A"))
    assert leads[:3] == ["II", "III", "aVF"]
    assert all(marked[ld] == "Элевация ST" for ld in ("II", "III", "aVF"))


def test_a_normal_record_keeps_the_default_leads_unmarked():
    assert finding_leads(run_pipeline(signal_from_case("normal"), mode="A")) == ([], {})


def test_a_gated_photo_marks_nothing():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    sig.provenance.notes = [SCALE_CONFLICT_NOTE]
    assert finding_leads(run_pipeline(sig, mode="A")) == ([], {})


def test_the_waveform_title_is_in_words_not_codes(tmp_path):
    from cardiolens.outputs.waveform import render_waveform

    svg = (tmp_path / "w.svg")
    render_waveform(run_pipeline(signal_from_case("stemi"), mode="A"), str(svg))
    text = svg.read_text()
    assert "st_elevation" not in text


@pytest.fixture
def clients(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "a:key-a,b:key-b")
    return TestClient(api.app, headers={"X-API-Key": "key-a"}), TestClient(api.app, headers={"X-API-Key": "key-b"})


def _analysis(client, case="stemi"):
    return client.post("/api/analyze", data={"case": case, "mode": "A"}).json()["analysis_id"]


def test_the_owner_gets_a_link_that_opens_the_pdf_without_a_key(clients):
    a, _ = clients
    aid = _analysis(a)
    r = a.post(f"/api/analysis/{aid}/share")
    assert r.status_code == 200
    url = r.json()["url"]
    assert url.startswith("/s/") and r.json()["expires_at"] > time.time() + 6 * 86400
    pdf = TestClient(api.app).get(url)          # no key
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"


def test_another_clinic_cannot_make_a_link(clients):
    a, b = clients
    assert b.post(f"/api/analysis/{_analysis(a)}/share").status_code == 404


def test_a_forged_or_expired_token_is_refused(clients):
    a, _ = clients
    aid = _analysis(a)
    good = a.post(f"/api/analysis/{aid}/share").json()["url"][3:]
    other_id = _analysis(a, "normal")
    forged = ".".join([other_id, *good.split(".")[1:]])            # another id, the same signature
    expired = api.share_token(aid, int(time.time()) - 10)
    anon = TestClient(api.app)
    assert anon.get(f"/s/{forged}").status_code == 404
    assert anon.get(f"/s/{expired}").status_code == 404
    assert anon.get("/s/garbage").status_code == 404


def test_the_link_carries_no_patient_data(clients):
    a, _ = clients
    aid = _analysis(a)
    token = a.post(f"/api/analysis/{aid}/share").json()["url"][3:]
    rid, exp, sig = token.split(".")
    assert rid == aid and exp.isdigit() and len(sig) == 32 and all(c in "0123456789abcdef" for c in sig)


def test_a_gone_analysis_is_said_not_500(clients):
    a, _ = clients
    aid = _analysis(a)
    url = a.post(f"/api/analysis/{aid}/share").json()["url"]
    service._STORE.pop(aid)
    r = TestClient(api.app).get(url)
    assert r.status_code == 410 and "больше не хранится" in r.text


def test_the_shared_pdf_leaves_the_photo_out(clients, monkeypatch):
    a, _ = clients
    aid = _analysis(a)
    service._STORE[aid].signal.provenance.overlay_jpeg = b"\xff\xd8not-a-real-jpeg"
    seen = {}
    real = api.write_pdf

    def spy(result, path, **kw):
        seen.update(kw)
        return real(result, path, **kw)
    monkeypatch.setattr(api, "write_pdf", spy)
    url = a.post(f"/api/analysis/{aid}/share").json()["url"]
    assert TestClient(api.app).get(url).status_code == 200 and seen == {"include_photo": False}
