"""D193: no comparison with or against a gated record; a missing prior is said, not skipped silently."""

from fastapi.testclient import TestClient

import cardiolens.pipeline as pl
from cardiolens.api.app import app
from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import SCALE_CONFLICT_NOTE
from cardiolens.serial import compare


def _fs(*codes):
    return FindingSet(findings=[Finding(code=c, label_ru=c, label_uz=c, confidence=0.8) for c in codes])


def _good(prior=None, **kw):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, **kw)
    if prior is not None:
        sig.context.prior_ecg_id = "prior1"
    return pl.run_pipeline(sig, mode="A", prior=prior)


def _gated(prior=None):
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    sig.provenance.notes = [SCALE_CONFLICT_NOTE]
    if prior is not None:
        sig.context.prior_ecg_id = "prior1"
    return pl.run_pipeline(sig, mode="A", prior=prior)


def _no_serial_alerts(res):
    return not any(a.code.startswith("serial_") for a in res.safety.alerts)


def test_a_gated_current_photo_is_not_compared():
    prior = _good()
    prior.findings.findings.append(Finding(code="st_elevation", label_ru="ST", label_uz="ST", confidence=0.9))
    res = _gated(prior)
    assert "photo_unreliable" in res.findings.codes()
    assert res.serial is not None and not res.serial.deltas and res.serial.qtc_delta_ms is None
    assert "текущий снимок" in res.serial.skipped_ru and res.serial.skipped_uz
    assert _no_serial_alerts(res)


def test_a_gated_prior_photo_is_not_compared():
    prior = _gated()
    res = _good(prior)
    assert res.serial is not None and not res.serial.deltas
    assert "прошлый снимок" in res.serial.skipped_ru
    assert _no_serial_alerts(res)


def test_two_good_records_are_still_compared():
    prior, res = _good(), None
    res = _good(prior)
    assert res.serial is not None and res.serial.skipped_ru is None and res.serial.hr_delta_bpm is not None


def test_a_resolved_finding_is_named_not_coded():
    c = compare(_fs("sinus_rhythm"), Measurements(), _fs("sinus_rhythm", "atrial_fibrillation"), Measurements())
    d = next(d for d in c.deltas if d.direction == "resolved")
    assert "фибрилляция предсердий" in d.label_ru and "atrial_fibrillation" not in d.label_ru
    assert "fibrillyatsiyasi" in d.label_uz and "atrial_fibrillation" not in d.label_uz


def test_an_unknown_prior_is_said_not_found(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "tester:test-key")
    client = TestClient(app, headers={"X-API-Key": "test-key"})
    r = client.post("/api/analyze", data={"case": "normal", "mode": "A", "prior_ecg_id": "nope00000000"})
    assert r.status_code == 200
    sc = r.json()["serial_comparison"]
    assert sc is not None and "не найден" in sc["skipped_ru"] and not sc["deltas"]


def test_another_clinics_prior_reads_as_not_found(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "a:key-a,b:key-b")
    a = TestClient(app, headers={"X-API-Key": "key-a"})
    b = TestClient(app, headers={"X-API-Key": "key-b"})
    prior_id = a.post("/api/analyze", data={"case": "normal", "mode": "A"}).json()["analysis_id"]
    sc_b = b.post("/api/analyze", data={"case": "normal", "mode": "A", "prior_ecg_id": prior_id}).json()
    assert "не найден" in sc_b["serial_comparison"]["skipped_ru"]
    sc_a = a.post("/api/analyze", data={"case": "normal", "mode": "A", "prior_ecg_id": prior_id}).json()
    assert sc_a["serial_comparison"]["skipped_ru"] is None
