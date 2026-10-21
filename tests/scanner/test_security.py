"""D98: keys on every API, limits before parsing, masked and encrypted images, scoped access."""

import struct
import time
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cardiolens.api.app import app

cv2 = pytest.importorskip("cv2")
AUTH = {"X-API-Key": "test-key"}


def _png_header_only(w: int, h: int) -> bytes:
    """A valid PNG that declares w x h px in a few dozen bytes (a decompression bomb's header)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b"")


# --- API keys -----------------------------------------------------------------------

def test_cardiolens_api_needs_a_key(monkeypatch):
    open_client = TestClient(app)
    assert open_client.post("/api/analyze", data={"case": "normal"}).status_code == 401
    assert open_client.post("/api/analyze", data={"case": "normal"}, headers={"X-API-Key": "nope"}).status_code == 401
    assert open_client.get("/api/analysis/abc/report.pdf").status_code == 401
    assert open_client.get("/health").status_code == 200 and open_client.get("/").status_code == 200
    monkeypatch.delenv("ECGSCAN_API_KEYS")
    assert open_client.post("/api/analyze", data={"case": "normal"}).status_code == 503     # never open


def test_the_key_is_checked_before_the_body_is_read():
    """FastAPI parses the form before dependencies: a 50-GB body would be spooled to disk."""
    hits = []

    def body():
        hits.append(1)
        yield b"x" * 1024

    r = TestClient(app).post("/scans", content=body(), headers={"Content-Type": "multipart/form-data; boundary=b"})
    assert r.status_code == 401 and not hits


def test_a_body_over_the_limit_is_refused_up_front():
    r = TestClient(app).post("/api/analyze", headers={**AUTH, "Content-Length": str(10**10)}, content=b"")
    assert r.status_code == 413


def test_an_analysis_is_seen_only_by_its_key(monkeypatch):
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic_a:key-a,clinic_b:key-b")
    a, b = TestClient(app, headers={"X-API-Key": "key-a"}), TestClient(app, headers={"X-API-Key": "key-b"})
    aid = a.post("/api/analyze", data={"case": "normal"}).json()["analysis_id"]
    assert a.get(f"/api/analysis/{aid}/conclusion").status_code == 200
    assert b.get(f"/api/analysis/{aid}/conclusion").status_code == 404
    assert b.get(f"/api/analysis/{aid}/report.pdf").status_code == 404


# --- images ---------------------------------------------------------------------------

def test_a_decompression_bomb_is_refused_before_decoding():
    from ecgscan.pipeline import scan

    bomb = _png_header_only(32000, 32000)
    assert len(bomb) < 200
    t0 = time.perf_counter()
    res = scan(bomb)
    assert res.status == "rejected" and res.reject_reasons == ["IMAGE_TOO_LARGE"]
    assert time.perf_counter() - t0 < 2


@pytest.mark.parametrize(("data", "code"), [(b"", "NOT_AN_ECG"), (b"\x89PNG junk", "NOT_AN_ECG")])
def test_empty_or_junk_bytes_are_rejected_not_crashed(data, code):
    from ecgscan.pipeline import scan

    res = scan(data)
    assert res.status == "rejected" and res.reject_reasons == [code]


def test_tiny_or_sliver_images_are_low_resolution():
    from ecgscan.pipeline import scan

    for shape in ((1, 1, 3), (2, 5000, 3)):
        res = scan(cv2.imencode(".png", np.full(shape, 255, np.uint8))[1].tobytes())
        assert res.status == "rejected" and res.reject_reasons == ["LOW_RESOLUTION"]


def test_header_and_footer_are_blank_in_saved_images(tmp_path):
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.sources import derive_limb_leads
    from ecgscan.layout import get_layout
    from ecgscan.pipeline import ScanHints, scan

    s = generate_synthetic_ecg(heart_rate_bpm=70, seed=2)
    mv = derive_limb_leads({n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names})
    img, gt = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=True))
    x0, y0, x1, y1 = gt.header_box_px
    assert (img[y0:y1, x0:x1] < 100).any()                         # the header has printed text
    res = scan(cv2.imencode(".png", img)[1].tobytes(), hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10),
               artifacts_dir=tmp_path)
    assert res.status != "rejected"
    first_row = min(t.baseline_px for t in gt.traces)
    for name in ("rectified.png", "overlay.png"):
        art = cv2.imread(str(tmp_path / name))
        band = art[: int(first_row - 2.5 * 10 * 10)]                 # 25 mm above the first row
        assert band.size and band.min() == 255, name


# --- scan service: queue, retention, comments ---------------------------------------------

@pytest.fixture
def svc_factory(tmp_path, result_factory):
    from ecgscan_service.service import ScanService, Settings

    made = []

    def make(scan_fn, **over):
        s = dict(root=tmp_path / "svc", keep_originals=False, retention_days=30, max_bytes=10**6,
                 mime=("image/png",), workers=1, feedback_path=tmp_path / "fb.jsonl", public_url=None,
                 purge_interval_s=0)
        s.update(over)
        svc = ScanService(Settings(**s), scan_fn=scan_fn, vault=False)
        made.append(svc)
        return svc

    yield make
    for svc in made:
        svc.close()


PNG = cv2.imencode(".png", np.full((40, 60, 3), 255, np.uint8))[1].tobytes()


def test_the_queue_is_bounded(svc_factory, result_factory):
    import threading

    from ecgscan_service.service import ScanError

    gate = threading.Event()

    def slow(data, *, hints, artifacts_dir):
        gate.wait(10)
        return result_factory(status="rejected", reject_reasons=["GRID_NOT_FOUND"])

    svc = svc_factory(slow, max_pending=2)
    svc.submit(PNG, owner="bot")
    svc.submit(PNG, owner="bot")
    with pytest.raises(ScanError) as exc:
        svc.submit(PNG, owner="bot")
    assert exc.value.code == 429
    gate.set()


def test_a_bomb_is_refused_at_submission(svc_factory, result_factory):
    from ecgscan_service.service import ScanError

    svc = svc_factory(lambda *a, **k: result_factory())
    with pytest.raises(ScanError) as exc:
        svc.submit(_png_header_only(32000, 32000), owner="bot")
    assert exc.value.code == 413


def test_retention_runs_on_a_timer_and_takes_the_comments(svc_factory, result_factory, tmp_path):
    import json

    def fake(data, *, hints, artifacts_dir):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(artifacts_dir / "overlay.png"), np.zeros((4, 4, 3), np.uint8))
        res = result_factory(status="needs_confirmation")
        res.artifacts = {"trace_overlay": str(artifacts_dir / "overlay.png")}
        return res

    svc = svc_factory(fake, retention_days=1e-6, purge_interval_s=0.2)
    sid = svc.submit(PNG, owner="bot")
    svc.wait(sid)
    svc.confirm(sid, actor="bot", decision="retake", comment="Ivanov I.I., re-check")
    assert (svc.artifacts_root / sid).exists()
    deadline = time.time() + 10
    while (svc.artifacts_root / sid).exists() and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(0.3)
    assert not (svc.artifacts_root / sid).exists()
    assert "Ivanov" not in (tmp_path / "fb.jsonl").read_text(encoding="utf-8")
    assert all("Ivanov" not in json.dumps(e) for e in svc.store.audit_log(sid))


def test_long_comments_are_refused(svc_factory, result_factory):
    from ecgscan_service.service import ScanError

    svc = svc_factory(lambda *a, **k: result_factory(status="needs_confirmation"))
    sid = svc.submit(PNG, owner="bot")
    svc.wait(sid)
    with pytest.raises(ScanError) as exc:
        svc.confirm(sid, actor="bot", decision="retake", comment="x" * 501)
    assert exc.value.code == 422


def test_bot_answers_only_in_private_chats():
    from ecgscan_bot.logic import BotConfig, allowed

    cfg = BotConfig(api_url="http://t", max_mb=20, poll_interval_s=1, poll_timeout_s=1, send_overlay=False,
                    public_url=None, allowed={111}, api_key="k")
    assert allowed(cfg, 111, "private") and not allowed(cfg, 111, "group") and not allowed(cfg, 111, "supergroup")


# --- DICOM listener ---------------------------------------------------------------------

def test_listener_needs_an_api_key(monkeypatch):
    pytest.importorskip("pynetdicom")
    from cardiolens.dicom_listener import ListenerConfig, api_analyzer

    monkeypatch.delenv("CARDIOLENS_LISTENER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CARDIOLENS_LISTENER_API_KEY"):
        api_analyzer(ListenerConfig(allowed_calling_aes=["X"]))


@pytest.mark.parametrize("limit", ["address", "size"])
def test_listener_cuts_off_unlisted_addresses_and_oversize_sends(tmp_path, limit):
    pytest.importorskip("pynetdicom")
    from cardiolens.dicom_listener import EcgListener, ListenerConfig
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from dicom_builder import dataset, group
    from test_dicom_listener import _free_port, _send

    got = []
    sig = generate_synthetic_ecg(heart_rate_bpm=70, seed=11)
    over = {"allowed_ips": ["10.9.9.9"]} if limit == "address" else {"max_bytes": 20_000, "max_pdu_bytes": 16384}
    cfg = ListenerConfig(port=_free_port(), allowed_calling_aes=["TELEECG01"], log_path=str(tmp_path / "rx.jsonl"),
                         **over)
    lst = EcgListener(cfg, analyze=lambda data: got.append(data) or {"analysis_id": "x"})
    lst.start()
    try:
        status = _send(cfg.port, dataset([group(sig.signal, sig.lead_names, sig.sampling_rate_hz)]))
    finally:
        lst.stop()
    assert status is None and got == []
    assert "refused" in lst.log_path.read_text()


def test_a_dark_photo_is_read_after_light_correction_and_sent_for_confirmation():
    """SD-062: indoor light (paper at grey ~105) used to be refused on layout / traces."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.sources import derive_limb_leads
    from ecgscan.io import flat_field, paper_level
    from ecgscan.layout import get_layout
    from ecgscan.pipeline import ScanHints, scan

    s = generate_synthetic_ecg(heart_rate_bpm=70, seed=6)
    mv = derive_limb_leads({n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names})
    img, gt = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=12.0, header=False))
    h, w = img.shape[:2]
    light = np.linspace(0.35, 0.5, w)[None, :, None] * np.ones((h, 1, 1))     # dim, uneven light
    dark = np.clip(img * light, 0, 255).astype(np.uint8)
    assert paper_level(dark) < 180 and paper_level(flat_field(dark)) > 220
    res = scan(cv2.imencode(".png", dark)[1].tobytes(), hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
    assert res.status == "needs_confirmation" and "LOW_LIGHT" in res.warnings
    from ecgscan.bench.metrics import compare_trace

    rs = []
    for tr in gt.traces:
        if tr.kind == "panel":
            out = res.leads[tr.lead]
            c = compare_trace(tr.samples_mv, tr.start_s, out.array(), out.start_s, 500.0, max_lag_s=0.25,
                              min_overlap=0.5, snr_cap_db=40.0)
            rs.append(c["r"] if c else 0.0)
    assert np.median(rs) > 0.95, rs
