"""D199: analyses survive a restart, encrypted on disk, 30 days, owner kept."""

import importlib
import os
import time

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from cardiolens.api import persist, service

api = importlib.import_module("cardiolens.api.app")


@pytest.fixture
def disk(monkeypatch, tmp_path):
    monkeypatch.setenv("ECGSCAN_STORAGE_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("CARDIOLENS_ANALYSES_DIR", str(tmp_path))
    monkeypatch.setenv("ECGSCAN_API_KEYS", "a:key-a,b:key-b")
    return tmp_path


def _restart():
    service._STORE.clear()
    service._OWNERS.clear()


def test_an_analysis_its_pdf_link_and_comparison_survive_a_restart(disk):
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    aid = a.post("/api/analyze", data={"case": "stemi", "mode": "A"}).json()["analysis_id"]
    link = a.post(f"/api/analysis/{aid}/share").json()["url"]
    _restart()
    assert a.get(f"/api/analysis/{aid}/report.pdf").status_code == 200
    assert TestClient(api.app).get(link).status_code == 200
    _restart()
    sc = a.post("/api/analyze", data={"case": "normal", "mode": "A", "prior_ecg_id": aid}).json()["serial_comparison"]
    assert sc["skipped_ru"] is None


def test_nothing_readable_is_on_disk(disk):
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    aid = a.post("/api/analyze", data={"case": "stemi", "mode": "A", "age_years": "61", "sex": "male"}).json()["analysis_id"]
    raw = (disk / f"{aid}.bin").read_bytes()
    # The file is a Fernet token: base64 text. Short words (b"male") turn up in random base64 by chance
    # (a flaky CI failure, 09.11), so the token's bytes are checked; a longer plaintext in the text too.
    import base64

    token = base64.urlsafe_b64decode(raw.strip() + b"=" * (-len(raw.strip()) % 4))
    for plain in (b"st_elevation", "Элевация".encode(), b"male", b"61.0", b"key-a"):
        assert plain not in token
    for plain in (b"st_elevation", "Элевация".encode(), b"key-a"):
        assert plain not in raw
    assert oct(os.stat(disk / f"{aid}.bin").st_mode)[-3:] == "600"


def test_another_clinic_cannot_read_it_after_a_restart(disk):
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    b = TestClient(api.app, headers={"X-API-Key": "key-b"})
    aid = a.post("/api/analyze", data={"case": "normal", "mode": "A"}).json()["analysis_id"]
    _restart()
    assert b.get(f"/api/analysis/{aid}/report.pdf").status_code == 404
    assert a.get(f"/api/analysis/{aid}/report.pdf").status_code == 200


def test_older_than_30_days_is_not_read_and_is_removed(disk, monkeypatch):
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    aid = a.post("/api/analyze", data={"case": "normal", "mode": "A"}).json()["analysis_id"]
    _restart()
    later = time.time() + 31 * 86400
    monkeypatch.setattr(time, "time", lambda: later)
    assert persist.load(aid) is None
    assert persist.purge(now=later) == 1 and not (disk / f"{aid}.bin").exists()


def test_without_a_storage_key_nothing_is_written(monkeypatch, tmp_path):
    monkeypatch.delenv("ECGSCAN_STORAGE_KEY", raising=False)
    monkeypatch.setenv("CARDIOLENS_ANALYSES_DIR", str(tmp_path))
    monkeypatch.setenv("ECGSCAN_API_KEYS", "a:key-a")
    TestClient(api.app, headers={"X-API-Key": "key-a"}).post("/api/analyze", data={"case": "normal", "mode": "A"})
    assert not list(tmp_path.iterdir())
    assert TestClient(api.app).get("/health").json()["analyses_kept_on_disk"] is False


def test_an_altered_file_reads_as_not_found(disk):
    a = TestClient(api.app, headers={"X-API-Key": "key-a"})
    aid = a.post("/api/analyze", data={"case": "normal", "mode": "A"}).json()["analysis_id"]
    p = disk / f"{aid}.bin"
    data = bytearray(p.read_bytes())
    data[40] ^= 0xFF
    p.write_bytes(bytes(data))
    _restart()
    assert a.get(f"/api/analysis/{aid}/report.pdf").status_code == 404


def test_a_path_like_id_is_never_read(disk):
    assert persist.load("../../etc/passwd") is None and persist.load("ABCDEF123456") is None
