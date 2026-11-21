"""The phone probe page for the Telegram Mini App capture (SD-134) must not send or load anything."""

import re
from pathlib import Path

import yaml

PAGE = Path(__file__).resolve().parents[2] / "tools/scanner/miniapp_probe.html"

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
    cfg = yaml.safe_load((PAGE.parents[2] / "configs/scanner/pipeline.yaml").read_text(encoding="utf-8"))
    full = float(re.search(r"FULL_READ_PX_MM = ([0-9.]+)", src).group(1))
    assert full == cfg["quality"]["min_px_per_mm"]
