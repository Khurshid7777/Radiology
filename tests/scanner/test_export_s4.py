"""S4 (TZ 18.10, 18.14): the read signal exported as the contract JSON, CSV and WFDB, and read
back without loss."""

import json

import numpy as np
import pytest

from ecgscan.contract import ScanResult, Trace
from ecgscan.export import channels, export, read_csv, read_wfdb_zip, to_csv, to_wfdb_zip

pytest.importorskip("wfdb")
FS = 500.0
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _page(result_factory):
    """A 3x4+rhythm read: each lead 2.5 s in its column, gaps in some, a 10-s rhythm II."""
    rng = np.random.default_rng(0)
    leads = {}
    for i, lead in enumerate(LEADS):
        x = np.sin(np.arange(1250) / 37 + i) * rng.uniform(0.3, 2.5) + rng.normal(0, 0.01, 1250)
        x[100 + 7 * i:130 + 7 * i] = np.nan                # a gap the page did not show
        leads[lead] = Trace.from_array(x, start_s=2.5 * (i // 3), fs=FS, lead=lead, quality=0.9)
    rhythm = [Trace.from_array(np.cos(np.arange(5000) / 41) * 1.7, start_s=0.0, fs=FS, lead="II", quality=0.8)]
    return result_factory(leads=leads, rhythm=rhythm, status="needs_confirmation", warnings=["TRACE_GAPS"])


def test_one_axis_standard_order_rhythm_after(result_factory):
    names, data = channels(_page(result_factory))
    assert names == [*LEADS, "II_rhythm"] and data.shape == (13, 5000)
    assert np.isnan(data[0, 1250:]).all() and np.isfinite(data[3, 1250:1250 + 100]).all()   # I: 0-2.5 s, aVR: 2.5-5 s


def test_csv_reads_back_bit_for_bit(result_factory):
    res = _page(result_factory)
    names, data = channels(res)
    n2, t, d2 = read_csv(to_csv(res))
    assert n2 == names and t[1] == pytest.approx(1 / FS) and len(t) == data.shape[1]
    assert np.array_equal(np.isnan(d2), np.isnan(data)) and np.array_equal(d2[~np.isnan(d2)], data[~np.isnan(data)])


def test_wfdb_reads_back_bit_for_bit(result_factory):
    """Format 32 at 100 000 adu/mV is the contract's own precision (5 decimals of mV)."""
    res = _page(result_factory)
    names, data = channels(res)
    n2, fs, d2, comments = read_wfdb_zip(to_wfdb_zip(res))
    assert n2 == names and fs == FS
    assert np.array_equal(np.isnan(d2), np.isnan(data)) and np.array_equal(d2[~np.isnan(d2)], data[~np.isnan(data)])
    assert any("status needs_confirmation" in c for c in comments) and any("TRACE_GAPS" in c for c in comments)


def test_the_contract_json_reads_back_as_the_same_result(result_factory):
    res = _page(result_factory)
    data, media, ext = export(res, "json")
    back = ScanResult.model_validate(json.loads(data))
    assert media == "application/json" and ext == "json" and back == res


def test_an_unknown_format_is_refused(result_factory):
    with pytest.raises(ValueError, match="format must be one of"):
        export(_page(result_factory), "edf")


def test_the_api_exports_a_scan_and_audits_it(tmp_path, monkeypatch, result_factory):
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient

    from ecgscan_service.app import app
    from ecgscan_service.router import KEYS_ENV, get_service
    from ecgscan_service.service import ScanService, Settings
    from ecgscan_service.vault import KEY_ENV

    monkeypatch.setenv(KEYS_ENV, "dr:secret-1,other:secret-2")
    monkeypatch.setenv(KEY_ENV, Fernet.generate_key().decode())
    page = _page(result_factory)

    def fake(data, *, hints, artifacts_dir):
        return page

    svc = ScanService(Settings(root=tmp_path / "svc", keep_originals=False, retention_days=30, max_bytes=1_000_000,
                               mime=("image/png",), workers=1, feedback_path=tmp_path / "fb.jsonl", public_url=None),
                      scan_fn=fake)
    app.dependency_overrides[get_service] = lambda: svc
    try:
        http = TestClient(app)
        import cv2

        png = cv2.imencode(".png", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()
        sid = http.post("/scans", files={"file": ("ecg.png", png, "image/png")},
                        headers={"X-API-Key": "secret-1"}).json()["scan_id"]
        svc.wait(sid)
        r = http.get(f"/scans/{sid}/export?format=wfdb", headers={"X-API-Key": "secret-1"})
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert f'filename="ecg-{sid[:8]}.zip"' in r.headers["content-disposition"]
        names, _, back, _ = read_wfdb_zip(r.content)
        assert names == channels(page)[0] and back.shape == channels(page)[1].shape
        r = http.get(f"/scans/{sid}/export?format=csv", headers={"X-API-Key": "secret-1"})
        assert r.status_code == 200 and r.text.startswith("time_s,I,II,III")
        assert http.get(f"/scans/{sid}/export?format=edf", headers={"X-API-Key": "secret-1"}).status_code == 422
        assert http.get(f"/scans/{sid}/export", headers={"X-API-Key": "secret-2"}).status_code == 404   # not theirs
        assert [a["details"] for a in svc.store.audit_log(sid) if a["action"] == "exported"] == \
            [{"format": "wfdb"}, {"format": "csv"}]
    finally:
        app.dependency_overrides.clear()
        svc.close()
