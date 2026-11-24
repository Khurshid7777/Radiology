"""D345: the site serves the Mini App probe page (SD-136) without a key and keeps everything else closed."""
import pytest
from fastapi.testclient import TestClient

from cardiolens.api.app import app


def test_probe_page_is_open_static_and_cannot_reach_the_network(monkeypatch):
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret")
    c = TestClient(app)
    r = c.get("/miniapp/probe")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "connect-src 'none'" in csp and "form-action 'none'" in csp
    assert r.headers["cache-control"] == "no-store"
    for word in ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "<script src", "tgWebAppData"):
        assert word not in r.text


def test_the_open_page_opens_nothing_else(monkeypatch):
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret")
    c = TestClient(app)
    assert c.get("/scans/config/capture-qc").status_code == 401
    assert c.get("/api/quality/summary").status_code == 401
    assert c.get("/miniapp/other").status_code == 404
    assert c.post("/miniapp/probe").status_code == 405


def test_capture_page_upload_and_send_on_the_site_d348(monkeypatch):
    """D348: the capture page is open; its upload and send need a valid ticket and read no body without one."""
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret")
    c = TestClient(app)
    r = c.get("/miniapp/capture?lang=ru")
    assert r.status_code == 200 and "connect-src 'self'" in r.headers["content-security-policy"]
    assert c.post("/miniapp/upload", content=b"\xff\xd8\xff" + b"0" * 100).status_code == 401
    assert c.post("/miniapp/upload", content=b"x", headers={"X-Miniapp-Ticket": "a" * 24 + ".1.0" + "b" * 63}).status_code == 401
    assert c.post("/miniapp/send").status_code == 401
    assert c.get("/scans/miniapp/inbox/" + "a" * 24).status_code == 401
    from ecgscan_service.miniapp import issue_ticket
    ticket, tid, _ = issue_ticket("secret", 5)
    h = {"X-Miniapp-Ticket": ticket}
    assert c.post("/miniapp/upload", content=b"not a photo at all", headers=h).status_code == 415
    assert c.post("/miniapp/send", headers=h).status_code == 409                      # nothing uploaded yet
    assert c.get(f"/scans/miniapp/inbox/{tid}", headers={"X-API-Key": "secret"}).status_code == 204


def test_the_ticket_never_reaches_the_access_log_d348():
    import logging

    from cardiolens.api.app import MiniappQueryFilter
    f = MiniappQueryFilter()
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", "/miniapp/capture?lang=ru&t=abc.123.def", "1.1", 200), None)
    assert f.filter(rec) and "t=" not in rec.getMessage() and "/miniapp/capture" in rec.getMessage()
    other = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                              ("1.2.3.4:5", "GET", "/api/x?days=7", "1.1", 200), None)
    assert f.filter(other) and "days=7" in other.getMessage()
    assert any(isinstance(x, MiniappQueryFilter) for x in logging.getLogger("uvicorn.access").filters)


def _declared_png(w: int, h: int) -> bytes:
    """A tiny PNG whose header declares ``w`` x ``h`` px (the pixel data is a stub: only the header is read)."""
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b""))


def test_a_decompression_bomb_is_refused_before_decoding_and_agent_keys_sign_no_ticket_d348(monkeypatch):
    """D348 review: a sub-MB PNG declaring 30000 x 30000 px decoded to ~5 GB in the upload's frame check; the agents'
    read-only key (D260) could sign a capture ticket."""
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret,agents:agentsecret")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret,agents:agentsecret")
    import cv2

    from ecgscan_service.miniapp import issue_ticket, turn_photo
    decoded = []
    real = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: decoded.append(1) or real(*a, **k))
    c = TestClient(app)
    bomb = _declared_png(30000, 30000)
    assert len(bomb) < 1000
    ticket, _, _ = issue_ticket("secret", 5)
    r = c.post("/miniapp/upload", content=bomb, headers={"X-Miniapp-Ticket": ticket})
    assert r.status_code == 413 and not decoded
    with pytest.raises(ValueError):
        turn_photo(bomb, 90, 95, 25 * 1024 * 1024)
    assert not decoded
    agent_ticket, _, _ = issue_ticket("agentsecret", 5)
    assert c.post("/miniapp/upload", content=bomb, headers={"X-Miniapp-Ticket": agent_ticket}).status_code == 401
    assert c.post("/miniapp/send", headers={"X-Miniapp-Ticket": agent_ticket}).status_code == 401


def _palette_tiff_without_colormap(w: int, h: int) -> bytes:
    """A TIFF PIL cannot open (palette, no colour map) but OpenCV's libtiff decodes — the review's bypass."""
    import struct
    import zlib

    data = zlib.compress(b"\x00" * w * 8)          # a stub strip: only the header matters for the refusal
    entries = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 1, 8), (259, 3, 1, 8), (262, 3, 1, 3), (273, 4, 1, 8 + 2 + 12 * 9 + 4),
               (277, 3, 1, 1), (278, 4, 1, h), (279, 4, 1, len(data))]
    ifd = struct.pack("<H", len(entries)) + b"".join(
        struct.pack("<HHI", t, k, n) + (struct.pack("<HH", v, 0) if k == 3 else struct.pack("<I", v)) for t, k, n, v in entries)
    return b"II*\x00" + struct.pack("<I", 8) + ifd + struct.pack("<I", 0) + data


def test_a_header_pil_cannot_read_is_never_decoded_and_opencv_has_its_own_ceiling_d348(monkeypatch):
    monkeypatch.setenv("CARDIOLENS_API_KEYS", "clinic:secret")
    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:secret")
    import cv2

    from ecgscan.io import image_size
    from ecgscan_service.miniapp import issue_ticket, size_refusal, turn_photo
    tif = _palette_tiff_without_colormap(20000, 20000)
    assert len(tif) < 100_000 and image_size(tif) is None
    assert size_refusal(tif) == "UNREADABLE_HEADER"          # the scanner's form of the refusal (SD, merged in D349)
    decoded = []
    real = cv2.imdecode
    monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: decoded.append(1) or real(*a, **k))
    ticket, _, _ = issue_ticket("secret", 5)
    assert TestClient(app).post("/miniapp/upload", content=tif, headers={"X-Miniapp-Ticket": ticket}).status_code == 415
    with pytest.raises(ValueError):
        turn_photo(tif, 90, 95, 25 * 1024 * 1024)
    assert not decoded
    from pathlib import Path
    dockerfile = (Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile").read_text()
    assert "ENV OPENCV_IO_MAX_IMAGE_PIXELS=60000000" in dockerfile
    from ecgscan_service.router import _pool
    assert _pool()._max_workers == 2


def test_the_site_page_works_as_a_telegram_mini_app_d349():
    """D349: the page tells Telegram it is ready and opens full height; the signed Telegram data is never read; in a
    browser nothing changes; no key travels in an address."""
    import re
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "src" / "cardiolens" / "api" / "static" / "index.html").read_text()
    block = page[page.index("// D349:"):page.index("function esc(")]
    assert "web_app_ready" in block and "web_app_expand" in block and "TelegramWebviewProxy" in block
    assert "postMessage(body, TG_WEB)" in block and "const TG_WEB = 'https://web.telegram.org'" in block
    assert "tgWebAppData" not in page.replace("(tgWebAppData)", "")            # named in a comment only, never read
    assert "if (tgApp)" in block                                               # everything else only inside Telegram
    assert not re.search(r"postMessage\([^)]*['\"]\*['\"]", page)              # never to any origin
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "x-frame-options" not in {k.lower() for k in r.headers}   # Telegram Web may frame it
    assert r.headers["content-security-policy"] == "frame-ancestors 'self' https://web.telegram.org"   # nobody else may
    keyblock = page[page.index("const KEY = 'cardiolens_key';"):page.index("function keyStatus()")]
    assert "inTelegram ? null" in keyblock and "confirm(" in keyblock              # review: no silent key replacement
