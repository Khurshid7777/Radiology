"""SD-150: lead labels printed on a strip photo (I, II, III, aVR, aVL, aVF, V1-V6), read by the project's OCR engine.

Only the twelve lead names are kept; any other text (the header, numbers, a misread) is dropped. A label read as something
else is worse than one not read (it would name a group wrongly), so the matching is strict: no guessing of "14" for "V4".
Not part of the scan yet (SD-149 decides how it is used, after the owner's samples).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
GROUPS = (("I", "II", "III"), ("aVR", "aVL", "aVF"), ("V1", "V2", "V3"), ("V4", "V5", "V6"))
_TOKEN = re.compile(r"(?<![A-Za-z0-9])(aV[RLF]|AV[RLF]|avr|avl|avf|V\s?[1-6]|v[1-6]|III|II|I)(?![A-Za-z0-9])")


@dataclass(frozen=True)
class Label:
    lead: str
    x: float          # centre of the text box, photo px
    y: float
    score: float


def normalise(token: str) -> str | None:
    """The lead a token names, or None."""
    t = token.replace(" ", "")
    if t.upper() in ("AVR", "AVL", "AVF"):
        return "aV" + t[-1].upper()
    if t.upper() in ("V1", "V2", "V3", "V4", "V5", "V6"):
        return t.upper()
    if t in ("I", "II", "III"):
        return t
    return None


def leads_in(text: str) -> list[str]:
    """The lead names in one OCR text, in order; nothing else is guessed."""
    out = []
    for m in _TOKEN.finditer(text):
        lead = normalise(m.group(1))
        if lead:
            out.append(lead)
    return out


def _views(bgr: np.ndarray, *, upscale: float, tile_px: int, whole_px: int) -> list[tuple[np.ndarray, float, int, int]]:
    """What the OCR sees: (image, its scale, x0, y0 in photo px). The detector reduces a large image to ~1000 px, and a
    2-3 mm label is then a few pixels: a small photo is enlarged, a large one is read in overlapping tiles at its size."""
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) * upscale <= whole_px:       # a part of a strip (~1200 px): read whole, enlarged
        big = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC) if upscale != 1 else bgr
        return [(big, upscale, 0, 0)]
    step = int(0.85 * tile_px)
    out = []
    for y0 in range(0, max(1, h - tile_px // 6), step):
        for x0 in range(0, max(1, w - tile_px // 6), step):
            out.append((bgr[y0:y0 + tile_px, x0:x0 + tile_px], 1.0, x0, y0))
    return out


def read_lead_labels(bgr: np.ndarray, *, min_score: float = 0.5, upscale: float = 2.0, tile_px: int = 1100,
                     whole_px: int = 2600, engine=None) -> list[Label]:
    """The lead labels on the photo, each with where it is (photo px). A small photo is enlarged ``upscale`` times (the
    labels are 2-3 mm tall; at 2x the detector finds aVL, V2, V3 it missed at 1x, SD-150); a large one is read in tiles
    of ``tile_px``. A label read twice in overlapping tiles is kept once. ``engine``: an OCR callable (RapidOCR)."""
    if engine is None:
        from ecgscan.calibration.header_ocr import _engine

        engine = _engine()
    out: list[Label] = []
    for img, k, x0, y0 in _views(bgr, upscale=upscale, tile_px=tile_px, whole_px=whole_px):
        res, _ = engine(img)
        for r in res or []:
            box, text, score = r[0], r[1], float(r[2])
            if box is None or score < min_score:
                continue
            names = leads_in(text)
            if len(names) != 1:          # one label per box: "aVR aVL" in one box cannot be placed name by name
                continue
            x = x0 + float(np.mean([p[0] for p in box])) / k
            y = y0 + float(np.mean([p[1] for p in box])) / k
            if any(o.lead == names[0] and abs(o.x - x) < 15 and abs(o.y - y) < 15 for o in out):
                continue
            out.append(Label(names[0], x, y, score))
    return out


def groups_seen(labels: list[Label], *, column_tol_px: float) -> list[tuple[tuple[str, str, str], float]]:
    """The strip groups the labels name: (group, its column x), when at least two of its three labels are read and stand
    in one column (within ``column_tol_px``) in their row order, top to bottom. A group is never named by one label."""
    out = []
    for group in GROUPS:
        found = [next((lb for lb in labels if lb.lead == lead), None) for lead in group]
        got = [lb for lb in found if lb is not None]
        if len(got) < 2:
            continue
        xs = [lb.x for lb in got]
        if max(xs) - min(xs) > column_tol_px:
            continue
        ys = [lb.y for lb in found if lb is not None]
        if ys != sorted(ys):                  # I above II above III
            continue
        out.append((group, float(np.median(xs))))
    return sorted(out, key=lambda g: g[1])
