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
    assert src.count("https://web.telegram.org") == 1 and src.count("postMessage(") == 1   # Telegram Web's frame only
    assert not re.search(r"https?://|\bsrc\s*=\s*[\"'](?!\")", src.replace("https://web.telegram.org", ""))
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
    assert csp.endswith("frame-ancestors https://web.telegram.org")  # only Telegram Web may frame the page
    assert "http" not in csp.replace("https://web.telegram.org", "") and "*" not in csp   # no outside source of any kind
    assert "img-src" not in csp and "media-src" not in csp           # nothing the page does not use
    head = client.head("/miniapp/probe")
    assert head.status_code == 200 and head.headers["content-security-policy"] == csp
    assert r.headers["cache-control"] == "no-store" and r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["x-content-type-options"] == "nosniff"
    # The scanner's own routes stay behind the key, and nothing else lives under /miniapp.
    assert client.get("/scans/config/capture-qc").status_code == 401
    assert client.get("/miniapp/").status_code == 404 and client.get("/miniapp/probe/x").status_code == 404
    assert client.post("/miniapp/probe").status_code == 405


def test_probe_says_whether_the_live_frame_will_go_green_on_a_whole_sheet():
    """SD-160: the verdict uses the live frame's own numbers (capture_qc min_px_per_mm, LIVE_FRAME.md fill)."""
    import importlib.util

    src = PAGE.read_text(encoding="utf-8")
    qc = yaml.safe_load((ROOT / "configs/scanner/capture_qc.yaml").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("calc", ROOT / "tools/scanner/live_frame_calc.py")
    calc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc)
    assert float(re.search(r"LIVE_PX_MM = ([0-9.]+)", src).group(1)) == qc["thresholds"]["min_px_per_mm"]
    assert float(re.search(r"LIVE_FILL = ([0-9.]+)", src).group(1)) == calc.FILL
    assert "getCapabilities" in src and "liveVerdict(photoLong, 'полное фото')" in src
    assert "лист A4 целиком пройдёт" in src and "только частями" in src
