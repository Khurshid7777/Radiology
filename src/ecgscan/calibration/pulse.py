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


def _line_phase(profile: np.ndarray, period: float, *, min_mean: float = 0.25, min_ratio: float = 4.0) -> float | None:
    """Phase of lines every ``period`` px in an ink profile (mean ink per column or row), or
    ``None``: no such lines (the usual case: the grid is coloured or too light for the mask)."""
    if profile.size < 3 * period:
        return None
    bins = max(4, int(round(period)))
    idx = ((np.arange(profile.size) % period) / period * bins).astype(int) % bins
    fold = np.bincount(idx, weights=profile, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    k = int(np.argmax(fold))
    if fold[k] < min_mean or fold[k] < min_ratio * max(float(np.median(fold)), 1e-6):
        return None
    return (k + 0.5) / bins * period


def grid_in_mask(mask: np.ndarray, px_per_mm: float) -> bool:
    """The 5-mm grid's vertical lines are in the ink mask (SD-079)."""
    return _line_phase(mask.mean(axis=0), 5.0 * float(px_per_mm)) is not None


def _grid_lattice(mask: np.ndarray, ppm: float, tol_mm: float = 1.0, gray: np.ndarray | None = None,
                  darker: float = 1.15, darker_min: int = 20):
    """A test "is this pair a grid cell" (both edges on 5-mm grid lines, or its top a grid
    line), when the grid's vertical lines got into the ink mask (a grey or dark grid, a compressed copy: the owner's CardioTouch
    strip, SD-079); ``None`` when they did not. The mask breaks those lines where the
    horizontal ones cross, and each piece pairs with the next into a "pulse" one cell wide.

    With the page's ``gray``, a pair whose edges are clearly darker than the grid's lines is
    a pulse drawn on them, not a cell (SD-084: ECG-Image-Kit draws its pulse on the 5-mm lines,
    trace grey 0-29 against grid 76): darkness (255 - grey) ``darker`` times the lines' and
    ``darker_min`` grey levels below them. Each edge is compared with its own line just above
    the pair (below it at the page's top): a page-wide median of the lines' pixels in the mask
    was mostly the pulses' legs and the trace's crossings on a page whose light grid is barely
    in the mask (Kit 12390_hr, SD-084)."""
    period = 5.0 * ppm
    px = _line_phase(mask.mean(axis=0), period)
    if px is None:
        return None
    tol = max(2.0, tol_mm * ppm)

    def on(v: int) -> bool:
        d = (v - px) % period
        return min(d, period - d) <= tol

    side = max(2, int(round(1.5 * ppm)))
    gap = max(1, int(round(0.5 * ppm)))
    band = max(1, int(round(0.3 * ppm)))

    def ruled(xa: int, xb: int, top: int) -> bool:
        """The "flat top" goes on past both edges: a horizontal grid line, not a pulse top."""
        rows = mask[max(0, top - band):top + band + 1]
        left, right = rows[:, max(0, xa - side):max(0, xa - gap)], rows[:, xb + gap:xb + side]
        return (left.size > 0 and right.size > 0 and left.any(axis=0).mean() > 0.8
                and right.any(axis=0).mean() > 0.8)

    half = max(1, int(round(0.3 * ppm)))

    near, far = max(1, int(round(1.0 * ppm))), max(2, int(round(6.0 * ppm)))

    def grey(x: int, y0: int, y1: int) -> float | None:
        """The darkest pixel across a column band, row by row, median; ``None`` if no rows."""
        y0, y1 = max(0, y0), min(gray.shape[0], y1)
        if y1 - y0 < near:
            return None
        return float(np.median(gray[y0:y1, max(0, x - half):x + half + 1].min(axis=1)))

    def darker_than_its_line(x: int, top: int, bottom: int) -> bool:
        edge = grey(x, top, bottom)
        line = grey(x, top - far, top - near)
        if line is None:
            line = grey(x, bottom + near, bottom + far)
        if edge is None or line is None:
            return False
        dark, line_dark = 255.0 - edge, 255.0 - line
        return dark >= darker * line_dark and dark - line_dark >= darker_min

    def cell(xa: int, xb: int, top: int, bottom: int) -> bool:
        if not ((on(xa) and on(xb)) or ruled(xa, xb, top)):
            return False
        if gray is None:
            return True
        return not (darker_than_its_line(xa, top, bottom) and darker_than_its_line(xb, top, bottom))

    return cell


def find_pulses(mask: np.ndarray, px_per_mm: float, *, min_h_mm: float = 3.0, max_h_mm: float = 25.0,
                min_w_mm: float = 2.0, max_w_mm: float = 15.0, edge_tol_mm: float = 0.6,
                top_coverage: float = 0.85, gray: np.ndarray | None = None) -> list[Pulse]:
    """Pulses on the ink ``mask``; ``gray`` (the page, optional) tells a pulse drawn on the
    grid's lines from a cell of them when those lines are in the mask (SD-084)."""
    ppm = float(px_per_mm)
    strokes = _vertical_strokes(mask, int(min_h_mm * ppm), int(max_h_mm * ppm))
    tol = max(2, int(round(edge_tol_mm * ppm)))
    lattice = _grid_lattice(mask, ppm, gray=gray)
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
            # A cell of a 5-mm grid that reached the ink mask (SD-079): both edges on its
            # lines, or a "flat top" that is a grid line going on past both edges.
            if lattice is not None and lattice(xa, xb, top, (ba + bb) // 2):
                continue
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


def extend_pulses(mask: np.ndarray, pulses: list[Pulse], px_per_mm: float, *, max_gap_mm: float = 4.0,
                  edge_tol_mm: float = 3.0, max_h_mm: float = 25.0) -> list[Pulse]:
    """A pulse whose two edges were broken by the mask (a compressed copy, a faded print: the
    owner's CardioTouch strip, a 20-mm pulse found as its top 5.2 mm, read as 5 mm/mV — D180)
    is extended down while BOTH edges go on, gaps up to ``max_gap_mm`` bridged, to the same
    level (``edge_tol_mm``: one edge may lose its last millimetres to the mask; the longer edge
    sets the bottom). Only found pulses are extended: no new ones are made, so a page whose
    pulses were right is unchanged unless both edges really continue."""
    ppm = float(px_per_mm)
    if not pulses:
        return pulses
    closed = widen_strokes(mask, max(1, int(round(0.4 * ppm))), max(1, int(round(max_gap_mm * ppm))))
    half = max(1, int(round(0.4 * ppm)))
    tol = max(2, int(round(edge_tol_mm * ppm)))
    h, w = mask.shape

    def run_end(x: int, y0: int) -> int:
        col = closed[:, max(0, x - half):min(w, x + half + 1)].any(axis=1)
        y = y0
        while y + 1 < h and col[y + 1]:
            y += 1
        return y

    out: list[Pulse] = []
    for p in pulses:
        mid = (p.top + p.bottom) // 2
        el, er = run_end(p.x_left, mid), run_end(p.x_right, mid)
        bottom = max(el, er)
        band = max(1, int(round(0.3 * ppm)))
        inner = mask[p.top + 2 * band:bottom - 2 * band, p.x_left + 2 * band:p.x_right - 2 * band]
        if (min(el, er) > p.bottom + tol and abs(el - er) <= tol and (bottom - p.top) / ppm <= max_h_mm
                and not (inner.size and (inner.mean(axis=0) > 0.8).any())):
            out.append(Pulse(p.x_left, p.x_right, p.top, bottom, round((bottom - p.top) / ppm, 3), p.width_mm))
        else:
            out.append(p)
    return out


def widen_strokes(mask: np.ndarray, widen_px: int, close_px: int) -> np.ndarray:
    """A gap-tolerant copy of a mask for the pulse search: widened ``widen_px`` across (a
    slanted line spread over neighbouring columns) and with vertical gaps up to ``close_px``
    closed (a line the mask broke into pieces)."""
    import cv2

    m = cv2.dilate(mask.astype(np.uint8), np.ones((1, max(1, widen_px)), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((max(1, close_px), 1), np.uint8)).astype(bool)
