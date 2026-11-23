"""The phone probe page for the Telegram Mini App capture (SD-134) must not send or load anything."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "src/ecgscan_service/static/miniapp_probe.html"

# Anything that could move a photo or a number off the phone, or pull code in.
FORBIDDEN = ("fetch(", "XMLHttpRequest", "FormData", "sendBeacon", "WebSocket", "EventSource", "<form", "<iframe",
             "<link", "import(", "localStorage", "sessionStorage", "document.cookie", "tgWebAppData", "initData")


def test_probe_page_has_no_network_code():
    src = PAGE.read_text(encoding="utf-8")
    for word in FORBIDDEN:
        assert word not in src, word
    # No address of any kind: nothing is loaded from outside, nothing can be posted to it.
    assert not re.search(r"https?://|\bsrc\s*=\s*[\"'](?!\")", src)
    assert "<script src" not in src


def test_probe_page_measures_the_three_capture_paths():
    src = PAGE.read_text(encoding="utf-8")
    assert "getUserMedia" in src and "takePhoto" in src                      # the camera inside the page
    assert 'capture="environment"' in src                                    # the phone's own camera
    assert src.count('type="file"') == 2                                     # ... and the gallery
    # The page's "enough for a full reading" line is the scanner's own threshold.
    cfg = yaml.safe_load((ROOT / "configs/scanner/pipeline.yaml").read_text(encoding="utf-8"))
    full = float(re.search(r"FULL_READ_PX_MM = ([0-9.]+)", src).group(1))
    assert full == cfg["quality"]["min_px_per_mm"]


def test_the_page_is_served_open_with_headers_that_forbid_any_network(monkeypatch):
    """SD-136: Telegram has no clinic key; the page is static and the browser itself is told to block every connection."""
    from fastapi.testclient import TestClient

    from ecgscan_service.app import app

    monkeypatch.setenv("ECGSCAN_API_KEYS", "clinic:k1")
    client = TestClient(app)
    r = client.get("/miniapp/probe")
    assert r.status_code == 200 and r.text == PAGE.read_text(encoding="utf-8")
    assert r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "connect-src 'none'" in csp and "form-action 'none'" in csp
    assert "http" not in csp and "*" not in csp                      # no outside source of any kind
    assert r.headers["cache-control"] == "no-store" and r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["x-content-type-options"] == "nosniff"
    # The scanner's own routes stay behind the key, and nothing else lives under /miniapp.
    assert client.get("/scans/config/capture-qc").status_code == 401
    assert client.get("/miniapp/").status_code == 404 and client.get("/miniapp/probe/x").status_code == 404
    assert client.post("/miniapp/probe").status_code == 405
