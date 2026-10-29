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

# Russian units as printed by CIS recorders ("50.0 мм/сек", "20 мм/мВ", SD-078) and as the
# Latin-script OCR model returns them: "MM/ceK", "MM/c", "MM/MB".
_MM = r"(?:mm|мм)"
SPEED_RE = re.compile(rf"(\d{{1,2}}(?:[.,]\d)?)\s*{_MM}\s*/\s*(?:sec|s|[cс][eе][kк]|[cс])(?![a-zа-я])", re.IGNORECASE)
GAIN_RE = re.compile(rf"(\d{{1,2}}(?:[.,]\d)?)\s*{_MM}\s*/\s*(?:m|м)(?:v|в|b)(?![a-zа-я])", re.IGNORECASE)


@dataclass
class HeaderReading:
    available: bool
    reason: str | None = None
    texts: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)


def _values(texts: list[str]) -> tuple[list[float], list[float]]:
    speeds, gains = [], []
    for t in texts:
        s = t.replace("O", "0").replace("o", "0") if any(c.isdigit() for c in t) else t
        speeds += [float(v.replace(",", ".")) for v in SPEED_RE.findall(s)]
        gains += [float(v.replace(",", ".")) for v in GAIN_RE.findall(s)]
    return speeds, gains


def parse_settings(texts: list[str]) -> tuple[list[float], list[float]]:
    """Speed (mm/s) and gain (mm/mV) values in OCR strings ('25mm/s10mm/mV' included).

    Newer RapidOCR (1.4) returns '25' and 'mm/s' as separate boxes: what single
    strings do not give is looked for in the boxes joined in reading order."""
    speeds, gains = _values(texts)
    if not speeds or not gains:
        j_speeds, j_gains = _values([" ".join(texts)])
        speeds = speeds or j_speeds
        gains = gains or j_gains
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


def tiles(width: int, band_h: int, *, tile_aspect: float = 3.0, min_tile_px: int = 0,
          overlap_px: int = 0) -> list[tuple[int, int]]:
    """The OCR tiles across a band: (x0, x1), each ``tile_aspect`` band heights wide (at least
    ``min_tile_px``), neighbours overlapping by a quarter tile or ``overlap_px`` (at most half a
    tile), whichever is more; the last one ends at the band's end."""
    step = max(64, int(band_h * tile_aspect), min_tile_px)
    advance = max(1, step - max(step // 4, min(overlap_px, step // 2)))
    out, x = [], 0
    while True:
        out.append((x, min(width, x + step)))
        if x + step >= width:
            return out
        x += advance


def _coloured(band: np.ndarray, frac: float, chroma: int = 60) -> bool:
    """At least ``frac`` of the band's pixels are clearly coloured (a red or pink grid)."""
    c = band.max(axis=2).astype(np.int16) - band.min(axis=2).astype(np.int16)
    return float((c > chroma).mean()) >= frac


def read_bands(bgr: np.ndarray, *, header_frac: float = 0.25, footer_frac: float = 0.12,
               tile_aspect: float = 3.0, min_band_px: int = 8, min_tile_px: int = 0,
               overlap_px: int = 0, colour_frac: float = 0.0) -> HeaderReading:
    """Speed and gain read from the header band, then the footer band, in overlapping ``tiles``:
    a setting as wide as the overlap is whole in some tile (Kit 12390: "25mm/s", 15 mm wide,
    was cut into "25mm/" and "mm/s" by a 9-mm overlap, SD-087). Bands that left speed or gain
    unread and are coloured (at least ``colour_frac`` of their pixels; 0 = never) are read
    again on their brightest channel (SD-088)."""
    ok, why = ocr_available()
    if not ok:
        return HeaderReading(False, why)
    h, w = bgr.shape[:2]
    bands = [bgr[: int(h * header_frac)], bgr[h - int(h * footer_frac):]]
    texts: list[str] = []

    def settled() -> bool:
        speeds, gains = parse_settings(texts)
        return bool(speeds) and bool(gains)

    def read(bands: list[np.ndarray]) -> None:
        for band in bands:
            bh = band.shape[0]
            if bh < min_band_px:
                continue
            if texts and settled():
                break                               # the header already gave speed and gain
            for x0, x1 in tiles(w, bh, tile_aspect=tile_aspect, min_tile_px=min_tile_px, overlap_px=overlap_px):
                res, _ = _engine()(band[:, x0:x1])
                new = [r[1] for r in (res or [])]
                texts.extend(new)
                # Speed and gain read: the rest of the band is not read either (as the footer is
                # not read after the header; ~0.3-0.5 s a tile, SD-069).
                if new and settled():
                    break

    read(bands)
    coloured = [b for b in bands if b.shape[0] >= min_band_px and b.ndim == 3 and _coloured(b, colour_frac)]
    if colour_frac > 0 and coloured and not settled():
        # Black text on a coloured grid: a red 5-mm line through the lowercase letters kept the
        # units off ("25", "10" read, "mm/s" not: Kit 12390). Read again on the brightest
        # channel, where red and pink go white and black stays black (SD-088).
        read([np.repeat(b.max(axis=2)[:, :, None], 3, axis=2) for b in coloured])
    speeds, gains = parse_settings(texts)
    return HeaderReading(True, None, texts, speeds, gains)
