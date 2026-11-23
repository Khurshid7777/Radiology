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
