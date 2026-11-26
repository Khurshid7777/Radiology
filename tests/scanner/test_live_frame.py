"""SD-156: the live camera frame ("green frame") on the Mini App capture page: the numbers behind it, the perspective
measure, the server's check of camera frames and the page."""

import importlib.util
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pytest

from ecgscan.i18n import strings
from ecgscan_service.miniapp import TICKET_HEADER, LiveGate, Ticket, issue_ticket

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/ecgscan_service/static/miniapp_capture.html"
FIX = Path(__file__).parent / "fixtures" / "capture_qc"


def _calc():
    spec = importlib.util.spec_from_file_location("live_frame_calc", ROOT / "tools/scanner/live_frame_calc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the numbers ------------------------------------------------------------------------------------

def test_the_numbers_behind_the_frame():
    c = _calc()
    assert c.max_object_mm(1920) == pytest.approx(216.0) and c.max_object_mm(3840) == pytest.approx(432.0)
    a4_1080 = c.shot_for(297.0, 1920)
    assert not a4_1080["enough"] and a4_1080["parts"] == 2 and a4_1080["px_per_mm"] == pytest.approx(5.82, abs=0.01)
    a4_4k = c.shot_for(297.0, 3840)
    assert a4_4k["enough"] and a4_4k["parts"] == 1 and 24 < a4_4k["distance_cm"] < 26
    assert c.visible_mm(c.distance_mm(330.0)) == pytest.approx(330.0)
    d = c.distance_mm(297.0 / c.FILL)
    # turned left-right the spacing goes with the square of the magnification: twice the up-down case
    assert c.yaw_spread(2.0, 297.0, d) == pytest.approx(2 * c.pitch_spread(2.0, 297.0, d), rel=0.02)
    assert c.yaw_spread(c.angle_for(0.08, 297.0, d), 297.0, d) == pytest.approx(0.08, abs=1e-6)
    assert 2.5 < c.angle_for(0.08, 297.0, d) < 3.1
    text = c.tables()
    assert "видео 3840×2160" in text and "мало → 2 части" in text and "8%" in text


# --- the perspective measure ------------------------------------------------------------------------

def _turned(img: np.ndarray, deg: float, axis: str) -> np.ndarray:
    """The sheet on a desk shot with the phone turned by ``deg`` about the vertical ("v") or horizontal ("h") axis."""
    h, w = img.shape[:2]
    c = cv2.copyMakeBorder(img, h // 8, h // 8, w // 8, w // 8, cv2.BORDER_CONSTANT, value=(90, 80, 70))
    hh, ww = c.shape[:2]
    f, t, out = 1.1 * ww, math.radians(deg), []
    for x, y in ((0, 0), (ww, 0), (ww, hh), (0, hh)):
        u, v = x - ww / 2, y - hh / 2
        x3, y3, z = (u * math.cos(t), v, f + u * math.sin(t)) if axis == "v" else (u, v * math.cos(t), f + v * math.sin(t))
        out.append((ww / 2 + f * x3 / z, hh / 2 + f * y3 / z))
    m = cv2.getPerspectiveTransform(np.float32([(0, 0), (ww, 0), (ww, hh), (0, hh)]), np.float32(out))
    return cv2.warpPerspective(c, m, (ww, hh), borderValue=(90, 80, 70))


def _spread(bgr: np.ndarray) -> float | None:
    from ecgscan.capture_qc import check_frame

    k = 1600 / bgr.shape[1]
    small = cv2.resize(bgr, (1600, round(bgr.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return check_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale_to_capture=1 / k).scale_spread


def test_the_spread_grows_with_the_angle_the_sheet_was_shot_at():
    sheet = cv2.imread(str(FIX / "clean_12.jpg"))
    sheet = cv2.resize(sheet, (sheet.shape[1] * 2, sheet.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    flat = _spread(_turned(sheet, 0, "v"))
    assert flat is not None and flat < 0.005
    one = _spread(_turned(sheet, 1, "v"))
    assert one is not None and 0.008 < one < 0.03
    assert _spread(_turned(sheet, 4, "v")) > 0.04
    assert _spread(_turned(sheet, 4, "h")) > _spread(_turned(sheet, 1, "h"))


# --- the gate ----------------------------------------------------------------------------------------

def test_the_gate_keeps_the_pace_the_count_and_one_check_at_a_time():
    g = LiveGate(min_interval_s=0.7, max_frames=3)
    a, b = Ticket("a" * 24, "bot", 2000), Ticket("b" * 24, "bot", 2000)
    assert g.enter(a, now=1000.0)
    assert not g.enter(b, now=1000.1)                      # another check runs: refused, never queued
    g.leave()
    assert g.enter(b, now=1000.2)
    g.leave()
    assert not g.enter(a, now=1000.5)                      # too fast for this ticket
    assert g.enter(a, now=1001.0)
    g.leave()
    assert g.enter(a, now=1002.0)
    g.leave()
    assert not g.enter(a, now=1003.0)                      # three frames were its share
    assert g.enter(Ticket("c" * 24, "bot", 2000), now=1003.0)
    g.leave()
    assert not g._seen.get("x") and g.enter(a, now=2001.0) is True   # the ticket ran out: forgotten (and refused at the door)


# --- the service -------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from ecgscan_service import router
    from ecgscan_service.app import app

    monkeypatch.setenv("ECGSCAN_API_KEYS", "bot:k-bot,other:k-other")
    monkeypatch.setenv("ECGSCAN_SERVICE_DIR", str(tmp_path / "service"))
    monkeypatch.chdir(tmp_path)
    router.get_inbox.cache_clear()
    router.get_live_gate.cache_clear()
    yield TestClient(app)
    router.get_inbox.cache_clear()
    router.get_live_gate.cache_clear()


def _frame() -> tuple[bytes, int]:
    ref = json.loads((FIX / "clean_12.json").read_text())
    data = (FIX / "clean_12.jpg").read_bytes()
    return data, round(1280 * ref["scale_to_capture"])


def _head() -> tuple[dict, str]:
    ticket, tid, _ = issue_ticket("k-bot", 5)
    return {TICKET_HEADER: ticket}, tid


def test_a_good_frame_is_green_for_the_photo_the_shot_will_be(client, tmp_path, caplog):
    from ecgscan_service import router

    caplog.set_level(logging.DEBUG)
    data, final = _frame()
    head, tid = _head()
    r = client.post(f"/miniapp/live?lang=ru&final_width={final}", content=data, headers=head)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ready"] and d["codes"] == [] and d["px_per_mm"] > 11 and d["scale_spread"] is not None
    assert len(d["corners"]) == 4 and all(0 <= v <= 1.05 for xy in d["corners"] for v in xy)
    router.get_live_gate.cache_clear()
    small = client.post("/miniapp/live?lang=ru&final_width=1280", content=data, headers=head).json()
    assert not small["ready"] and "closer" in small["codes"]               # the same frame, a small photo: closer
    assert small["hints"][0] == strings("ru")["qc.closer"]
    assert router.get_inbox().attempt(Ticket(tid, "bot", 0), 1)            # the frames cost the button no upload
    assert not any(p.is_file() for p in tmp_path.rglob("*"))               # nothing kept
    said = "\n".join(r.getMessage() for r in caplog.records if not r.name.startswith("httpx"))
    assert tid not in said and head[TICKET_HEADER] not in said


def test_the_live_check_refuses_what_it_should(client):
    from ecgscan_service import router

    data, final = _frame()
    assert client.post(f"/miniapp/live?final_width={final}", content=data).status_code == 401
    head, tid = _head()
    big = b"\xff\xd8\xff" + b"x" * (601 * 1024)
    assert client.post("/miniapp/live", content=big, headers=head).status_code == 413
    router.get_live_gate.cache_clear()
    assert client.post("/miniapp/live", content=b"GIF89a" + b"x" * 100, headers=head).status_code == 415
    router.get_live_gate.cache_clear()
    assert client.post(f"/miniapp/live?final_width={final}", content=data, headers=head).status_code == 200
    assert client.post(f"/miniapp/live?final_width={final}", content=data, headers=head).status_code == 429   # too fast
    router.get_live_gate.cache_clear()
    with router.get_inbox()._lock:
        router.get_inbox()._used[tid] = 10 ** 10                           # the bot took the record: the ticket is spent
    assert client.post(f"/miniapp/live?final_width={final}", content=data, headers=head).status_code == 401


def test_a_strip_in_parts_is_not_asked_for_the_whole_sheet(client):
    from ecgscan_service import router

    img = cv2.imread(str(FIX / "clean_12.jpg"))
    part = img[:, 300:]                                    # the sheet runs off the frame's left edge
    data = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    final = round(part.shape[1] * json.loads((FIX / "clean_12.json").read_text())["scale_to_capture"])
    head, _ = _head()
    whole = client.post(f"/miniapp/live?final_width={final}", content=data, headers=head).json()
    router.get_live_gate.cache_clear()
    inparts = client.post(f"/miniapp/live?final_width={final}&parts=1", content=data, headers=head).json()
    assert "frame" in whole["codes"] and "frame" not in inparts["codes"]


# --- the page ----------------------------------------------------------------------------------------

def test_the_page_offers_the_live_camera_only_where_the_page_may_use_it(client):
    page = client.get("/miniapp/capture?lang=uz")
    assert page.status_code == 200
    assert "connect-src 'self'" in page.headers["content-security-policy"]
    assert "camera=(self)" in page.headers["permissions-policy"]
    src = page.text
    assert "getUserMedia" in src and "liveSupported()" in src and "post('live'" in src
    assert "ImageCapture" in src and "takePhoto" in src and "track.stop" not in src and "t.stop()" in src
    blob = json.loads(re.search(r"const PAGE = (\{.*?\});\s*//", src, re.DOTALL).group(1))
    assert blob["live"]["frame_width"] == 1600 and blob["live"]["ready_frames"] == 2
    assert blob["texts"]["miniapp.live_open"] == strings("uz")["miniapp.live_open"]


@pytest.mark.parametrize("lang", ["ru", "uz"])
def test_the_live_texts_exist_in_both_languages(lang):
    s = strings(lang)
    for k in ("qc.grid", "qc.skew", "miniapp.live_open", "miniapp.live_close", "miniapp.live_shoot", "miniapp.live_wait",
              "miniapp.live_parts", "miniapp.live_moving", "miniapp.live_denied", "miniapp.live_lead"):
        assert s[k].strip(), k
    assert "{ppm}" in s["miniapp.live_ready"]


# --- SD-158: the upload check at 1600 px and the weak-grid hint ---------------------------------------

def _hd_a4_frame() -> bytes:
    """A 1080p camera frame of an A4 sheet across it: ~4.7 px/mm, the scanner refuses it."""
    src = cv2.imread(str(FIX / "clean_12.jpg"))
    k = min(0.9 * 1920 / src.shape[1], 0.9 * 1080 / src.shape[0])
    img = cv2.resize(src, (round(src.shape[1] * k), round(src.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((1080, 1920, 3), (70, 80, 90), np.uint8)
    y0, x0 = (1080 - img.shape[0]) // 2, (1920 - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def test_a_small_a4_frame_is_never_all_good_on_upload():
    from ecgscan_service.router import photo_hints

    got = photo_hints(_hd_a4_frame(), "ru")
    assert got["codes"] and {"closer", "grid"} & set(got["codes"])
    assert got["px_per_mm"] is None or got["px_per_mm"] < 8


def test_a_weak_grid_is_said_in_words(monkeypatch):
    from ecgscan.capture_qc import FrameQC
    from ecgscan_service import router

    weak = FrameQC(width=1600, height=1000, factor=1, brightness=200.0, glare=0.0, sharpness=100.0, tilt_deg=0.0,
                   period_px=30.0, px_per_mm=9.3, sheet=0.6, hints=[], ok=True, period_strength=0.1)
    monkeypatch.setattr("ecgscan.capture_qc.check_frame", lambda *a, **k: weak)
    data = cv2.imencode(".jpg", np.full((1000, 1600, 3), 230, np.uint8))[1].tobytes()
    for lang in ("ru", "uz"):
        got = router.photo_hints(data, lang)
        assert got["codes"] == ["grid"] and got["hints"] == [strings(lang)["qc.grid"]]
    monkeypatch.setattr("ecgscan.capture_qc.check_frame", lambda *a, **k: FrameQC(**{**weak.to_dict(), "period_strength": 0.3}))
    assert router.photo_hints(data, "ru")["codes"] == []
