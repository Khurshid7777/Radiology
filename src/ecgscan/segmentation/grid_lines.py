"""Grid lines taken back out of the trace mask (R1): the black 5-mm lattice (SD-099) and
light-red pixels (SD-101).

The 5-mm lattice (SD-099). A black-and-white scan, or a page printed with a black grid, has
grid lines as dark as the trace: the segmenter keeps both (ecgdig b/w scans: 14-16 "rows" on a
4-row page, LAYOUT_UNKNOWN). Only what is certainly a grid line goes:

* a straight run at least ``min_h_mm`` long (horizontal) or ``min_v_mm`` tall (vertical) —
  a trace is never that straight for that long, except a flat baseline;
* lying on the page's 5-mm lattice (its phase read from those runs) — so a flat baseline
  between two grid lines stays;
* outside the calibration pulses (their legs are drawn on grid lines, SD-084).

Where a trace crosses a removed line the line's pixels with trace on both sides are put
back, so the crossing leaves no gap. The pipeline uses it only when the grid's lines are in
the mask and no layout was found without it: a page that reads is never touched.
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


def drop_red(ink: np.ndarray, bgr: np.ndarray, *, max_redness: int = 60, min_level: int = 140,
             speckle_px: int = 0, pink_level: int | None = None, pink_redness: int = 20) -> np.ndarray:
    """``ink`` without light-red pixels (SD-101): red standing more than ``max_redness`` above
    green and blue, red itself above ``min_level``. A red grid line; never the trace, which is
    black, and stays dark (low red) even when a warm lamp tints it. The same array when
    nothing is dropped."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    gb = np.maximum(g, b)
    red = ((r - gb) > max_redness) & (r > min_level)
    if pink_level is not None:
        # The segmenter's mask is a little wider than a red line: the paper beside it, pale
        # pink, came along (grid crossings kept as 1-2-mm blobs). Light and reddish is paper; a
        # faded grey trace is light but not reddish, and stays.
        red |= ((r - gb) > pink_redness) & (np.minimum(g, b) > pink_level)
    hit = ink & red
    if not hit.any():
        return ink
    out = ink & ~red
    if speckle_px > 0:
        # Where two red lines cross the print is darker: dots the colour test leaves, ~0.5 mm
        # each, isolated. The trace reader jumped to one (II: a 1.9-mV "spike" on ecgdig
        # 1004-v1). Pieces no bigger than ``speckle_px`` either way go.
        _, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
        small = (st[:, cv2.CC_STAT_WIDTH] <= speckle_px) & (st[:, cv2.CC_STAT_HEIGHT] <= speckle_px)
        small[0] = False
        if small.any():
            out = out & ~small[lab]
    return out


def drop_labels(ink: np.ndarray, long_ink: np.ndarray, panels, px_per_mm: float, *, window_mm: float = 8.0,
                before_mm: float = 1.0, band_px: float = 150.0, far_mm: float = 0.0) -> np.ndarray:
    """``ink`` with only ``long_ink`` (the trace's long strokes) kept in a window at each panel's
    start: from ``before_mm`` before it to ``window_mm`` after, ``band_px`` either side of its
    baseline. A lead label printed there (SD-102) is a few short strokes, never long."""
    out = ink.copy()
    h, w = ink.shape
    if far_mm > 0 and long_ink.any():
        # Text anywhere (a footer "25mm/s" under the rhythm strip): pieces that are not the
        # trace's long strokes and stand more than ``far_mm`` off them. A piece of the trace
        # the mask broke off (a QRS tip) lies next to the rest.
        dist = cv2.distanceTransform((~long_ink).astype(np.uint8), cv2.DIST_L2, 3)
        rest = (out & ~long_ink).astype(np.uint8)
        n, lab, _, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        if n > 1:
            near = np.full(n, np.inf)
            np.minimum.at(near, lab[rest.astype(bool)], dist[rest.astype(bool)])
            far = near > far_mm * px_per_mm
            far[0] = False
            out &= ~far[lab]
    a, b = int(round(before_mm * px_per_mm)), int(round(window_mm * px_per_mm))
    for p in panels:
        x0, x1 = max(0, int(p.x0) - a), min(w, int(p.x0) + b)
        y0, y1 = max(0, int(p.baseline - band_px)), min(h, int(p.baseline + band_px))
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] &= long_ink[y0:y1, x0:x1]
    return out
