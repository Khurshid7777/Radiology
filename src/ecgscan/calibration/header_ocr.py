"""Paper speed and gain printed in the page header/footer, read by OCR (TZ §5.4, SD-013).

OCR runs on the header and footer bands only, never on the trace area. The
engine is RapidOCR (Apache-2.0, ONNX, from PyPI; Tesseract needs a system
package). When it is not installed the reader says so and the source is simply
absent (confidence 0), never guessed.

Wide pages are read in overlapping tiles: the detector downscales a long strip
and loses the small print (a 50 mm/s page is ~7000 px wide at 12 px/mm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

SPEED_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*mm\s*/\s*s(?![a-z])", re.IGNORECASE)
GAIN_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*mm\s*/\s*mv", re.IGNORECASE)


@dataclass
class HeaderReading:
    available: bool
    reason: str | None = None
    texts: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)


def parse_settings(texts: list[str]) -> tuple[list[float], list[float]]:
    """Speed (mm/s) and gain (mm/mV) values in OCR strings ('25mm/s10mm/mV' included)."""
    speeds, gains = [], []
    for t in texts:
        s = t.replace("O", "0").replace("o", "0") if any(c.isdigit() for c in t) else t
        speeds += [float(v.replace(",", ".")) for v in SPEED_RE.findall(s)]
        gains += [float(v.replace(",", ".")) for v in GAIN_RE.findall(s)]
    return speeds, gains


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_available() -> tuple[bool, str | None]:
    try:
        _engine()
    except Exception as exc:  # ImportError or a broken model install
        return False, f"OCR engine unavailable: {type(exc).__name__}"
    return True, None


def read_bands(bgr: np.ndarray, *, header_frac: float = 0.25, footer_frac: float = 0.12,
               tile_aspect: float = 3.0) -> HeaderReading:
    ok, why = ocr_available()
    if not ok:
        return HeaderReading(False, why)
    h, w = bgr.shape[:2]
    bands = [bgr[: int(h * header_frac)], bgr[h - int(h * footer_frac):]]
    texts: list[str] = []
    for band in bands:
        bh = band.shape[0]
        if bh < 8:
            continue
        step = max(64, int(bh * tile_aspect))
        x = 0
        while True:
            res, _ = _engine()(band[:, x:x + step])
            texts += [r[1] for r in (res or [])]
            if x + step >= w:
                break
            x += step * 3 // 4
    speeds, gains = parse_settings(texts)
    return HeaderReading(True, None, texts, speeds, gains)
