"""Calibration pulse detection on a rectified page (TZ §5.4).

A pulse is the "Π" a printer draws before (or after) a row: a rising stroke, a flat
top and a falling stroke back to the baseline. Its height in mm is the gain
(mm/mV for 1 mV); its width is a *weak* hint of the paper speed (SD-011: vendors
differ). Detection works in millimetres on the ink mask:

1. vertical ink strokes 3..25 mm tall (one per stroke, not per pixel column);
2. pairs of strokes 2..15 mm apart whose tops and bottoms agree;
3. the top between them is inked along >= ``top_coverage`` of the span, and no
   interior column carries a stroke of the full height (a QRS has no flat top).

A 20 mm/mV pulse reaches into the neighbouring row; nothing here assumes it stays
inside its own row (the legacy detector capped the height at 0.75 row pitch,
which is why gain 20 was never read in M1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pulse:
    x_left: int
    x_right: int
    top: int
    bottom: int
    height_mm: float
    width_mm: float


def ink_mask(bgr: np.ndarray, px_per_mm: float, *, ratio: float = 0.6, max_chroma: int = 60) -> np.ndarray:
    """Dark, neutral pixels: the trace and printed text, not the coloured grid."""
    import cv2

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(15, int(round(6 * px_per_mm)) | 1)
    local = cv2.blur(g, (k, k))
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    return (g < ratio * local) & (chroma < max_chroma)


def _vertical_strokes(mask: np.ndarray, min_len: int, max_len: int) -> list[tuple[int, int, int]]:
    """(x, top, bottom) of vertical runs, adjacent columns of one stroke merged."""
    w = mask.shape[1]
    m = mask.astype(np.int8)
    d = np.diff(np.vstack([np.zeros((1, w), np.int8), m, np.zeros((1, w), np.int8)]), axis=0)
    starts = np.argwhere(d.T == 1)     # (x, y)
    ends = np.argwhere(d.T == -1)
    runs = []
    for (x, y0), (_, y1) in zip(starts, ends, strict=True):
        if min_len <= y1 - y0 <= max_len:
            runs.append((int(x), int(y0), int(y1)))
    runs.sort()
    merged: list[list[int]] = []
    for x, y0, y1 in runs:
        for mrg in merged[-6:]:
            if x - mrg[0] <= 2 and abs(y0 - mrg[1]) <= 3 and abs(y1 - mrg[2]) <= 3:
                mrg[0], mrg[1], mrg[2] = x, min(mrg[1], y0), max(mrg[2], y1)
                mrg[3] += 1
                break
        else:
            merged.append([x, y0, y1, 1])
    return [(x, y0, y1) for x, y0, y1, _ in merged]


def find_pulses(mask: np.ndarray, px_per_mm: float, *, min_h_mm: float = 3.0, max_h_mm: float = 25.0,
                min_w_mm: float = 2.0, max_w_mm: float = 15.0, edge_tol_mm: float = 0.6,
                top_coverage: float = 0.85) -> list[Pulse]:
    ppm = float(px_per_mm)
    strokes = _vertical_strokes(mask, int(min_h_mm * ppm), int(max_h_mm * ppm))
    tol = max(2, int(round(edge_tol_mm * ppm)))
    band = max(1, int(round(0.3 * ppm)))
    out: list[Pulse] = []
    used: set[int] = set()
    for i, (xa, ta, ba) in enumerate(strokes):
        if i in used:
            continue
        for j in range(i + 1, len(strokes)):
            xb, tb, bb = strokes[j]
            dx = xb - xa
            if dx > max_w_mm * ppm:
                break
            if dx < min_w_mm * ppm or j in used or abs(ta - tb) > tol or abs(ba - bb) > tol:
                continue
            top = (ta + tb) // 2
            span = mask[max(0, top - band):top + band + 1, xa + 1:xb]
            if span.size == 0 or span.any(axis=0).mean() < top_coverage:
                continue
            bottom = (ba + bb) // 2
            # No interior column may hold a stroke of the full height (that is a waveform).
            inner = mask[top + 2 * band:bottom - 2 * band, xa + 2 * band:xb - 2 * band]
            if inner.size and (inner.mean(axis=0) > 0.8).any():
                continue
            out.append(Pulse(xa, xb, top, bottom, round((bottom - top) / ppm, 3), round(dx / ppm, 3)))
            used.update((i, j))
            break
    # A pulse inside a taller one is a piece of it (its edges broken where something crosses
    # them), not a pulse of its own: its bottom would add a row that is not there (M6, SD-054).
    return [p for p in out if not any(q is not p and q.height_mm > p.height_mm and q.x_left - tol <= p.x_left
                                      and p.x_right <= q.x_right + tol and q.top - tol <= p.top
                                      and p.bottom <= q.bottom + tol for q in out)]


def summarize_pulses(pulses: list[Pulse], *, agree: float = 0.08) -> dict | None:
    """Largest group of pulses whose heights agree: median height / width and count."""
    if not pulses:
        return None
    hs = np.array([p.height_mm for p in pulses])
    best = max(range(len(pulses)), key=lambda i: int((np.abs(hs - hs[i]) <= agree * hs[i]).sum()))
    grp = [p for p in pulses if abs(p.height_mm - hs[best]) <= agree * hs[best]]
    return {"height_mm": float(np.median([p.height_mm for p in grp])),
            "width_mm": float(np.median([p.width_mm for p in grp])), "n_agree": len(grp), "n_found": len(pulses)}


def widen_strokes(mask: np.ndarray, widen_px: int, close_px: int) -> np.ndarray:
    """A gap-tolerant copy of a mask for the pulse search: widened ``widen_px`` across (a
    slanted line spread over neighbouring columns) and with vertical gaps up to ``close_px``
    closed (a line the mask broke into pieces)."""
    import cv2

    m = cv2.dilate(mask.astype(np.uint8), np.ones((1, max(1, widen_px)), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((max(1, close_px), 1), np.uint8)).astype(bool)
