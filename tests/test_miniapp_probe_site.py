"""D345: the site serves the Mini App probe page (SD-136) without a key and keeps everything else closed."""
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
