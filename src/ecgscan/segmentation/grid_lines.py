"""The 5-mm grid lines taken back out of the trace mask (R1, SD-099).

A black-and-white scan, or a page printed with a black grid, has grid lines as dark as the
trace: the segmenter keeps both (ECG-Image-Kit's b/w scans: 16 "rows" found on a 4-row page,
LAYOUT_UNKNOWN). Only what is certainly a grid line goes:

* a straight run at least ``min_h_mm`` long (horizontal) or ``min_v_mm`` tall (vertical) —
  a trace is never that straight for that long, except a flat baseline;
* lying on the page's 5-mm lattice (its phase read from those runs) — so a flat baseline
  between two grid lines stays;
* outside the calibration pulses (their legs are drawn on grid lines, SD-084).

Where a trace crosses a removed line the line's pixels with trace on both sides are put
back, so the crossing leaves no gap. Off unless the grid's lines are in the mask (the usual
red grid never is): pages that already read are unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np


def _lattice(profile: np.ndarray, period: float, min_frac: float) -> np.ndarray | None:
    """Indices of the profile on a lattice of lines every ``period`` px, or ``None``."""
    if profile.size < 3 * period or profile.max() <= 0:
        return None
    bins = max(4, int(round(period)))
    idx = ((np.arange(profile.size) % period) / period * bins).astype(int) % bins
    fold = np.bincount(idx, weights=profile, minlength=bins) / np.maximum(np.bincount(idx, minlength=bins), 1)
    k = int(np.argmax(fold))
    if fold[k] < min_frac:
        return None
    on = np.zeros(bins, bool)
    on[k] = True
    for d in (-1, 1):                                  # a scanned line is 3-6 px thick: its whole width
        for i in range(1, max(2, bins // 6)):
            j = (k + d * i) % bins
            if fold[j] < 0.25 * fold[k]:
                break
            on[j] = True
    return on[idx]


def _restore(removed: np.ndarray, kept: np.ndarray, reach: int, axis: int) -> np.ndarray:
    """Removed pixels with kept ink within ``reach`` px on both sides across the line."""
    before = np.zeros_like(kept)
    after = np.zeros_like(kept)
    n = kept.shape[axis]
    for s in range(1, reach + 1):
        if s >= n:
            break
        if axis == 0:
            before[s:] |= kept[:-s]
            after[:-s] |= kept[s:]
        else:
            before[:, s:] |= kept[:, :-s]
            after[:, :-s] |= kept[:, s:]
    return removed & before & after


def drop_grid_lines(ink: np.ndarray, px_per_mm: float, *, keep_boxes=(), min_h_mm: float = 20.0,
                    min_v_mm: float = 8.0, min_frac: float = 0.02, reach_mm: float = 0.8,
                    slack_mm: float = 0.2, band_mm: float = 40.0,
                    bridge_mm: float = 1.5) -> tuple[np.ndarray, float]:
    """(ink without the 5-mm grid lines, fraction of the ink removed)."""
    ink = ink.astype(bool)
    period = 5.0 * float(px_per_mm)
    u8 = ink.astype(np.uint8)
    found = {0: np.zeros_like(ink), 1: np.zeros_like(ink)}
    reach = max(2, int(round(reach_mm * px_per_mm)))
    slack = max(1, int(round(slack_mm * px_per_mm)))
    for axis, length_mm in ((0, min_h_mm), (1, min_v_mm)):
        n = max(3, int(round(length_mm * px_per_mm)))
        kernel = np.ones((1, n) if axis == 0 else (n, 1), np.uint8)
        # A scan is never exactly square to the grid: a line drifts a pixel every ~200 px.
        # Thickened across first, it stays one run; only the ink under it is taken.
        across = np.ones((2 * slack + 1, 1) if axis == 0 else (1, 2 * slack + 1), np.uint8)
        # The segmenter often breaks a grid line into dashes (where the other lines cross):
        # gaps up to ``bridge_mm`` along it are closed first.
        b = max(1, int(round(bridge_mm * px_per_mm)))
        along = np.ones((1, b) if axis == 0 else (b, 1), np.uint8)
        thick = cv2.morphologyEx(cv2.dilate(u8, across), cv2.MORPH_CLOSE, along)
        runs = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel).astype(bool) & ink
        # Horizontal lines: a lattice of rows (profile over y); vertical: of columns. Read band
        # by band (``band_mm`` across the lines): a page off square by a fraction of a degree
        # moves a line a few pixels from one side to the other.
        size = runs.shape[1 - axis]
        step = max(n, int(round(band_mm * px_per_mm)))
        for a in range(0, size, step):
            part = runs[:, a:a + step] if axis == 0 else runs[a:a + step, :]
            on = _lattice(part.mean(axis=1 - axis), period, min_frac)
            if on is None:
                continue
            if axis == 0:
                found[0][:, a:a + step] = part & on[:, None]
            else:
                found[1][a:a + step, :] = part & on[None, :]
    for x0, y0, x1, y1 in keep_boxes:
        for m in found.values():
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    lines = found[0] | found[1]
    if not lines.any():
        return ink, 0.0
    kept = ink & ~lines
    # A trace crossing a horizontal line has ink above and below it; a vertical one, left and right.
    out = kept | _restore(found[0], kept, reach, axis=0) | _restore(found[1], kept, reach, axis=1)
    return out, float((ink & ~out).sum()) / max(1, int(ink.sum()))
