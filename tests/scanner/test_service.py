"""M4: the /scans API (TZ §4, §6.3, §9, §12; SD-004, SD-005, SD-008)."""

import json
import threading

import numpy as np
import pytest

from ecgscan.contract import Calibration, Layout, Trace

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cryptography")
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from ecgscan_service.app import app
from ecgscan_service.router import KEYS_ENV, get_service
from ecgscan_service.service import ScanError, ScanService, Settings
from ecgscan_service.vault import KEY_ENV

KEY = {"X-API-Key": "secret-1"}
PNG = cv2.imencode(".png", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()
make_result = None


@pytest.fixture(autouse=True)
def _results(result_factory):
    global make_result
    make_result = result_factory


class FakeScan:
    """Stands in for ecgscan.scan: returns a prepared status, records the hints it got."""

    def __init__(self, status="ok", warnings=(), speed=25.0):
        self.status, self.warnings, self.speed = status, list(warnings), speed
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, data, *, hints, artifacts_dir):
        self.gate.wait(10)
        self.calls.append(hints)
        speed = hints.paper_speed_mm_s or self.speed
        status = "ok" if hints.paper_speed_mm_s else self.status
        res = make_result(status=status, warnings=[] if status == "ok" else self.warnings,
                          calibration=Calibration(paper_speed_mm_s=speed, gain_mm_mv=10,
                                                  speed_source="user" if hints.paper_speed_mm_s else
                                                  ("header_ocr" if speed else None),
                                                  gain_source="calibration_pulse", confidence=0.9))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(artifacts_dir / "overlay.png"), np.zeros((4, 4, 3), np.uint8))
        res.artifacts = {"trace_overlay": str(artifacts_dir / "overlay.png")}
        return res


def _settings(tmp_path, **over):
    s = dict(root=tmp_path / "svc", keep_originals=True, retention_days=30, max_bytes=1_000_000,
             mime=("image/jpeg", "image/png", "image/tiff"), workers=1, feedback_path=tmp_path / "fb.jsonl",
             public_url=None)
    s.update(over)
    return Settings(**s)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv(KEYS_ENV, "dr_karimova:secret-1,bot:secret-2")
    monkeypatch.setenv(KEY_ENV, Fernet.generate_key().decode())


@pytest.fixture
def make_client(tmp_path, env):
    made = []

    def _make(fake=None, *, vault=True, **over):
        svc = ScanService(_settings(tmp_path, **over), scan_fn=fake or FakeScan(), vault=vault)
        app.dependency_overrides[get_service] = lambda: svc
        made.append(svc)
        return TestClient(app), svc

    yield _make
    app.dependency_overrides.clear()
    for svc in made:
        svc.close()


def _submit(http, data=PNG, **form):
    r = http.post("/scans", files={"file": ("ecg.png", data, "image/png")}, data=form, headers=KEY)
    assert r.status_code == 202, r.text
    return r.json()["scan_id"]


def test_api_never_runs_open(tmp_path, monkeypatch, make_client):
    client, _ = make_client()
    assert client.get("/scans/x", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/scans/x").status_code == 401
    monkeypatch.delenv(KEYS_ENV)
    assert client.get("/scans/x", headers=KEY).status_code == 503


def test_upload_is_checked(make_client):
    client, _ = make_client(max_bytes=1000)
    r = client.post("/scans", files={"file": ("a.txt", b"hello", "text/plain")}, headers=KEY)
    assert r.status_code == 415
    big = PNG + b"\0" * 2000
    assert client.post("/scans", files={"file": ("a.png", big, "image/png")}, headers=KEY).status_code == 413


def test_unknown_client_is_refused_at_submission(make_client):
    client, _ = make_client()
    r = client.post("/scans", files={"file": ("ecg.png", PNG, "image/png")}, data={"client": "telegram"},
                    headers=KEY)
    assert r.status_code == 422


def test_scan_is_processed_with_messages_and_audit(make_client):
    client, svc = make_client()
    sid = _submit(client, client="flutter")
    svc.wait(sid)
    body = client.get(f"/scans/{sid}", headers=KEY).json()
    assert body["status"] == "ok" and body["result"]["scan_id"] == sid
    assert body["messages"]["ru"]["status"].startswith("Готово") and body["messages"]["uz"]["status"]
    log = client.get(f"/scans/{sid}/audit", headers=KEY).json()
    assert log[0]["action"] == "submitted" and log[0]["actor"] == "dr_karimova"
    art = client.get(f"/scans/{sid}/artifacts/trace_overlay", headers=KEY)
    assert art.status_code == 200 and art.headers["content-type"] == "image/png"
    assert client.get(f"/scans/{sid}/artifacts/../../scans.sqlite", headers=KEY).status_code == 404


def test_original_is_kept_only_encrypted(make_client):
    client, svc = make_client()
    sid = _submit(client)
    svc.wait(sid)
    stored = (svc.settings.root / "originals" / f"{sid}.bin").read_bytes()
    assert PNG not in stored and svc.vault.get(sid) == PNG


def test_without_a_key_nothing_is_kept(make_client):
    client, svc = make_client(vault=None)
    sid = _submit(client)
    svc.wait(sid)
    assert svc.get(sid).status == "ok"
    assert not (svc.settings.root / "originals").exists()
    r = client.post(f"/scans/{sid}/confirm", json={"decision": "corrected", "paper_speed_mm_s": 50}, headers=KEY)
    assert r.status_code == 409


def test_speed_correction_reruns_the_scan_and_is_recorded(make_client, tmp_path):
    fake = FakeScan(status="needs_confirmation", warnings=["SPEED_UNCERTAIN"], speed=None)
    client, svc = make_client(fake)
    sid = _submit(client)
    svc.wait(sid)
    assert svc.get(sid).status == "needs_confirmation"
    r = client.post(f"/scans/{sid}/confirm", json={"decision": "accepted"}, headers=KEY)
    assert r.status_code == 422                      # speed not established: cannot be just accepted
    r = client.post(f"/scans/{sid}/confirm", json={"decision": "corrected", "paper_speed_mm_s": 50}, headers=KEY)
    assert r.status_code == 200
    row = svc.wait(sid)
    assert fake.calls[-1].paper_speed_mm_s == 50
    assert row.status == "ok" and row.result["calibration"]["speed_source"] == "user"
    conf = row.result["confirmation"]
    assert conf["by"] == "dr_karimova" and conf["decision"] == "corrected"
    assert conf["changes"]["after"] == {"paper_speed_mm_s": 50.0}
    fb = [json.loads(x) for x in (tmp_path / "fb.jsonl").read_text().splitlines()]
    assert fb[-1]["decision"] == "corrected" and fb[-1]["changes"]["before"]["paper_speed_mm_s"] is None
    assert [a["action"] for a in svc.store.audit_log(sid)] == ["submitted", "confirm:corrected"]


def test_accept_and_retake(make_client):
    client, svc = make_client(FakeScan(status="needs_confirmation", warnings=["LIMB_LEADS_INCONSISTENT"]))
    sid = _submit(client)
    svc.wait(sid)
    r = client.post(f"/scans/{sid}/confirm", json={"decision": "accepted", "comment": "трассы верны"}, headers=KEY)
    assert r.json()["status"] == "ok" and r.json()["result"]["confirmation"]["decision"] == "accepted"
    rej = make_result(status="rejected", reject_reasons=["GLARE"], leads={})
    client2, svc2 = make_client(lambda data, **kw: rej, root=svc.settings.root.parent / "svc2")
    sid2 = _submit(client2)
    svc2.wait(sid2)
    body = client2.get(f"/scans/{sid2}", headers=KEY).json()
    assert body["messages"]["ru"]["reasons"][0].startswith("Блик")
    assert client2.post(f"/scans/{sid2}/confirm", json={"decision": "accepted"}, headers=KEY).status_code == 409
    r = client2.post(f"/scans/{sid2}/confirm", json={"decision": "retake"}, headers=KEY)
    assert r.json()["result"]["confirmation"]["decision"] == "retake"


def test_pending_scans_survive_a_restart(tmp_path, env):
    fake = FakeScan()
    fake.gate.clear()                                # the first service never gets to run it
    s = _settings(tmp_path)
    first = ScanService(s, scan_fn=fake)
    sid = first.submit(PNG, owner="bot")
    first.store.update(sid, status="processing")    # as if the process died mid-scan
    second = ScanService(s, scan_fn=FakeScan())
    assert second.wait(sid).status == "ok"
    fake.gate.set()
    first.close()
    second.close()
    no_vault = ScanService(_settings(tmp_path / "b"), scan_fn=fake, vault=None)
    sid2 = no_vault.store.create("lost", owner="bot", sha256="0" * 64, hints={}) or "lost"
    again = ScanService(_settings(tmp_path / "b"), scan_fn=fake, vault=None)
    assert again.get(sid2).status == "failed" and "restart" in again.get(sid2).error
    no_vault.close()
    again.close()


def test_retention_purges_originals_and_images(make_client):
    from datetime import UTC, datetime, timedelta

    client, svc = make_client()
    sid = _submit(client)
    svc.wait(sid)
    assert svc.purge_expired(datetime.now(UTC) + timedelta(days=31)) == 1
    assert not svc.vault.has(sid)
    assert client.get(f"/scans/{sid}/artifacts/trace_overlay", headers=KEY).status_code == 410
    assert client.get(f"/scans/{sid}", headers=KEY).json()["result"] is not None   # the signal stays


def _page_result(synth_mv):
    """A 3x4+rhythm ScanResult cut from a digital record (each lead in its window)."""
    from ecgscan.layout import get_layout

    spec = get_layout("3x4+rhythm")
    leads = {}
    for row in spec.rows:
        for c, ld in enumerate(row):
            i0, i1 = int(c * spec.panel_s * 500), int((c + 1) * spec.panel_s * 500)
            leads[ld] = Trace.from_array(synth_mv[ld][i0:i1], start_s=c * spec.panel_s, fs=500.0, lead=ld,
                                         quality=0.95)
    rhythm = [Trace.from_array(synth_mv["II"], start_s=0.0, fs=500.0, lead="II", quality=0.95)]
    return make_result(leads=leads, rhythm=rhythm, layout=Layout(name="3x4+rhythm", confidence=1.0))


def test_ok_scan_goes_to_cardiolens_as_a_photo(make_client, synth_mv):
    res = _page_result(synth_mv)
    client, svc = make_client(lambda data, **kw: res.model_copy(deep=True))
    sid = _submit(client)
    svc.wait(sid)
    r = client.post(f"/scans/{sid}/analysis", json={"mode": "B"}, headers=KEY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "escalate"                # a photo is never interpreted autonomously
    assert "Получено по фото" in body["conclusion_ru"] and "fotosuratidan" in body["conclusion_uz"]
    assert client.get(f"/scans/{sid}", headers=KEY).json()["analysis_id"] == body["analysis_id"]


def test_unconfirmed_scan_is_not_analysed(make_client):
    client, svc = make_client(FakeScan(status="needs_confirmation", warnings=["SPEED_UNCERTAIN"], speed=None))
    sid = _submit(client)
    svc.wait(sid)
    assert client.post(f"/scans/{sid}/analysis", headers=KEY).status_code == 409


def test_router_is_mounted_in_cardiolens(env):
    from cardiolens.api.app import app as cl_app

    assert TestClient(cl_app).get("/scans/x").status_code == 401        # the scanner's auth, not a 404


def test_real_page_end_to_end(tmp_path, env, synth_mv):
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    svc = ScanService(_settings(tmp_path, max_bytes=25_000_000))
    try:
        sid = svc.submit(cv2.imencode(".png", img)[1].tobytes(), owner="bot",
                         hints={"paper_speed_mm_s": 25, "gain_mm_mv": 10})
        row = svc.wait(sid, timeout_s=120)
        assert row.status == "ok", row.result and row.result["warnings"]
        assert row.result["calibration"]["speed_source"] == "user"
        assert set(row.result["artifacts"]) == {"rectified_image", "trace_overlay", "trace_mask",
                                                "trace_overlay_preview"}
        data, media = svc.artifact(sid, "trace_overlay_preview", "bot")
        assert media == "image/jpeg" and data[:3] == b"\xff\xd8\xff" and len(data) < 1_000_000
        assert all("/" not in v for v in row.result["artifacts"].values())
        # D136: with a storage key the images are kept encrypted, never as plaintext files.
        on_disk = sorted(p.name for p in (svc.artifacts_root / sid).iterdir())
        assert on_disk and all(n.endswith(".enc") for n in on_disk)
        with pytest.raises(ScanError) as other:
            svc.artifact(sid, "trace_overlay_preview", "dr_karimova")      # another key's scan
        assert other.value.code == 404
    finally:
        svc.close()


def test_service_loads_its_models_at_start_by_default(tmp_path):
    """SD-056: the deployed service scans one synthetic page at start; tests built with
    their own Settings do not."""
    from ecgscan_service.service import Settings

    assert Settings.load().warm_up is True
    assert _settings(tmp_path).warm_up is False
