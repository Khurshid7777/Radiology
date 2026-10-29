"""Paper speed and gain printed in the page header/footer, read by OCR (TZ §5.4, SD-013).

OCR runs on the header and footer bands only, never on the trace area. The
engine is RapidOCR (Apache-2.0, ONNX, from PyPI) with the Cyrillic PP-OCRv5
recogniser from ``models/ocr`` (S3, SD-090: it beat RapidOCR's own recogniser and
Tesseract rus+eng+uzb on Russian, Uzbek and English headers). When it is not
available the reader says so and the source is simply absent (confidence 0),
never guessed.

Wide pages are read in overlapping tiles: the detector downscales a long strip
and loses the small print (a 50 mm/s page is ~7000 px wide at 12 px/mm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# Russian units as printed by CIS recorders ("50.0 мм/сек", "20 мм/мВ", SD-078) and as the
# Latin-script OCR model returns them: "MM/ceK", "MM/c", "MM/MB". The Cyrillic model mixes the
# scripts inside a word ("25Mм/C", "10 мm/mB"), splits a number ("1 0mm/mV") and reads the
# digit after a unit as a letter ("мм/мВО,5-35 Гц"), S3 (SD-090).
_NUM = r"(\d(?:\s?\d)?(?:[.,]\d)?)"
_MM = r"[mм]{2}"
_END = r"(?![a-np-zа-нп-яё])"                   # a letter may not follow, but O/о may (a misread 0)
SPEED_RE = re.compile(rf"{_NUM}\s*{_MM}\s*/\s*(?:sec|s|[cс][eе][kк]|[cс]){_END}", re.IGNORECASE)
# Two overlapping OCR boxes of one "25mm/s": "25mm/" and its tail "m/s" read as "n/s" (Kit
# 12390 on a red grid, SD-089). Only in the boxes joined in reading order: "mm/" then a repeated
# "m/" (or "n/") then "s". A bare "25mm/" with no tail is not a speed.
SPEED_SPLIT_RE = re.compile(rf"{_NUM}\s*{_MM}\s*/\s*[mnм]{{1,2}}\s*/\s*(?:s|[cс]){_END}", re.IGNORECASE)
GAIN_RE = re.compile(rf"{_NUM}\s*{_MM}\s*/\s*[mм][vвb]{_END}", re.IGNORECASE)
_ZERO = re.compile(r"(?<=\d)[OoОоQ]|[OoОоQ](?=\d)")    # O, о, Q next to a digit: a misread 0


def _number(v: str) -> float:
    return float(v.replace(" ", "").replace(",", "."))


@dataclass
class HeaderReading:
    available: bool
    reason: str | None = None
    texts: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    boxes: list = field(default_factory=list)       # OcrBox, page coordinates
    engine: str | None = None


def _values(texts: list[str], *, split: bool = False) -> tuple[list[float], list[float]]:
    speeds, gains = [], []
    for t in texts:
        s = _ZERO.sub("0", t)
        speeds += [_number(v) for v in SPEED_RE.findall(s)]
        if split and not speeds:
            speeds += [_number(v) for v in SPEED_SPLIT_RE.findall(s)]
        gains += [_number(v) for v in GAIN_RE.findall(s)]
    return speeds, gains


def parse_settings(texts: list[str]) -> tuple[list[float], list[float]]:
    """Speed (mm/s) and gain (mm/mV) values in OCR strings ('25mm/s10mm/mV' included).

    Newer RapidOCR (1.4) returns '25' and 'mm/s' as separate boxes: what single
    strings do not give is looked for in the boxes joined in reading order, where a speed
    split into two overlapping boxes ('25mm/', 'n/s') is also read (SD-089)."""
    speeds, gains = _values(texts)
    if not speeds or not gains:
        j_speeds, j_gains = _values([" ".join(texts)], split=True)
        speeds = speeds or j_speeds
        gains = gains or j_gains
    return speeds, gains


@dataclass
class OcrBox:
    text: str
    score: float
    x0: int
    y0: int
    x1: int
    y1: int


@lru_cache(maxsize=4)
def _engine(rec_model: str | None = None, rec_sha256: str | None = None):
    """RapidOCR's detector with its own recogniser, or with ``rec_model`` (an ONNX recogniser whose
    dictionary is in its metadata: the Cyrillic PP-OCRv5 one, S3). A configured model that is
    missing or not the one in its card is an error, never a silent fallback (as the segmenter)."""
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    if rec_model:
        import hashlib

        try:
            from rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
        except ImportError as exc:      # 1.3+ moved it: pyproject pins 1.2.3 (SD-098)
            raise ImportError("rapidocr-onnxruntime 1.2.x required for the Cyrillic recogniser "
                              "(pip install 'rapidocr-onnxruntime==1.2.3')") from exc

        path = Path(rec_model)
        if not path.is_file():
            raise FileNotFoundError(f"OCR recogniser {path.name} not found")
        if rec_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != rec_sha256:
            raise ValueError(f"OCR recogniser {path.name} does not match its sha256 (model_card.json)")
        ocr.text_recognizer = TextRecognizer({"model_path": str(path), "use_cuda": False,
                                              "rec_img_shape": [3, 48, 320], "rec_batch_num": 6})
    return ocr


def engine_name(rec_model: str | None = None) -> str:
    return "rapidocr:" + (Path(rec_model).stem if rec_model else "ch_PP-OCRv3_rec")


def ocr_available(rec_model: str | None = None, rec_sha256: str | None = None) -> tuple[bool, str | None]:
    try:
        _engine(rec_model, rec_sha256)
    except Exception as exc:  # ImportError, a broken install, a missing or altered model
        return False, f"OCR engine unavailable: {type(exc).__name__}: {exc}"
    return True, None


def group_lines(boxes: list[OcrBox]) -> list[str]:
    """Boxes (page coordinates) as text lines, top to bottom, each left to right. Boxes of the
    overlapping tiles that cover one another keep the longer reading."""
    kept: list[OcrBox] = []
    for b in sorted(boxes, key=lambda b: (-len(b.text.strip()), -b.score)):
        area = max(1, (b.x1 - b.x0) * (b.y1 - b.y0))
        if not any(max(0, min(b.x1, k.x1) - max(b.x0, k.x0)) * max(0, min(b.y1, k.y1) - max(b.y0, k.y0))
                   > 0.6 * min(area, max(1, (k.x1 - k.x0) * (k.y1 - k.y0))) for k in kept):
            kept.append(b)
    lines: list[list[OcrBox]] = []
    for b in sorted(kept, key=lambda b: (b.y0 + b.y1) / 2):
        cy, hh = (b.y0 + b.y1) / 2, max(1, b.y1 - b.y0)
        if lines:
            last = lines[-1]
            ly = sum((k.y0 + k.y1) / 2 for k in last) / len(last)
            if abs(cy - ly) <= 0.5 * max(hh, max(k.y1 - k.y0 for k in last)):
                last.append(b)
                continue
        lines.append([b])
    return ["  ".join(k.text.strip() for k in sorted(ln, key=lambda k: k.x0)) for ln in lines]


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
               overlap_px: int = 0, colour_frac: float = 0.0, rec_model: str | None = None,
               rec_sha256: str | None = None, whole_header: bool = False) -> HeaderReading:
    """Speed and gain read from the header band, then the footer band, in overlapping ``tiles``:
    a setting as wide as the overlap is whole in some tile (Kit 12390: "25mm/s", 15 mm wide,
    was cut into "25mm/" and "mm/s" by a 9-mm overlap, SD-087). Bands that left speed or gain
    unread and are coloured (at least ``colour_frac`` of their pixels; 0 = never) are read
    again on their brightest channel (SD-088). ``whole_header``: the header band is read to its
    end even after speed and gain (its other fields: the recorder's measurements, model, date;
    S3); the footer is still read only when the header left speed or gain out."""
    ok, why = ocr_available(rec_model, rec_sha256)
    if not ok:
        return HeaderReading(False, why)
    engine = _engine(rec_model, rec_sha256)
    h, w = bgr.shape[:2]
    top_h, foot_y = int(h * header_frac), h - int(h * footer_frac)
    bands = [(bgr[:top_h], 0), (bgr[foot_y:], foot_y)]
    texts: list[str] = []
    boxes: list[OcrBox] = []

    def settled() -> bool:
        speeds, gains = parse_settings(texts)
        return bool(speeds) and bool(gains)

    def read(bands: list[tuple[np.ndarray, int]]) -> None:
        for band, y_off in bands:
            bh = band.shape[0]
            if bh < min_band_px:
                continue
            header = y_off == 0
            if texts and settled() and not (header and whole_header):
                break                               # the header already gave speed and gain
            for x0, x1 in tiles(w, bh, tile_aspect=tile_aspect, min_tile_px=min_tile_px, overlap_px=overlap_px):
                res, _ = engine(band[:, x0:x1])
                for r in res or []:
                    texts.append(r[1])
                    if r[0] is not None:
                        xs, ys = [p[0] for p in r[0]], [p[1] for p in r[0]]
                        boxes.append(OcrBox(r[1], float(r[2]), int(min(xs)) + x0, int(min(ys)) + y_off,
                                            int(max(xs)) + x0, int(max(ys)) + y_off))
                # Speed and gain read: the rest of the band is not read either (as the footer is
                # not read after the header; ~0.3-0.5 s a tile, SD-069), unless the whole header is.
                if res and settled() and not (header and whole_header):
                    break

    read(bands)
    coloured = [(b, y) for b, y in bands if b.shape[0] >= min_band_px and b.ndim == 3 and _coloured(b, colour_frac)]
    if colour_frac > 0 and coloured and not settled():
        # Black text on a coloured grid: a red 5-mm line through the lowercase letters kept the
        # units off ("25", "10" read, "mm/s" not: Kit 12390). Read again on the brightest
        # channel, where red and pink go white and black stays black (SD-088).
        read([(np.repeat(b.max(axis=2)[:, :, None], 3, axis=2), y) for b, y in coloured])
    speeds, gains = parse_settings(texts)
    return HeaderReading(True, None, texts, speeds, gains, boxes, engine_name(rec_model))
