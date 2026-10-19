"""Trace rows on a rectified page (TZ §5.6): how many, and where their baselines are.

Only ink that belongs to *long* connected strokes counts (a trace runs for
centimetres; a printed word or a lead label is a few millimetres), so the header
text no longer counts as a row — the legacy counter read the header as a fifth or
sixth row (M1/M2). Baselines are the peaks of that ink's horizontal projection
(ink piles up on the flat segments), at least ``min_gap_mm`` apart.
"""

from __future__ import annotations

import numpy as np


def long_stroke_mask(ink: np.ndarray, px_per_mm: float, min_width_mm: float = 20.0,
                     min_height_mm: float = 3.0, *, bridge_mm: float = 0.0, min_piece_mm: float = 4.0,
                     exclude_boxes: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Components at least ``min_width_mm`` long with a vertical stroke of at least
    ``min_height_mm`` in some column: a trace has complexes; a paper edge, a shadow
    border or a ruled line is long and, even when tilted, thin in every column.

    ``bridge_mm`` > 0: a trace the mask broke into pieces (a real strip: 18 + 45 + 36 mm)
    is judged whole — pieces joined across gaps up to this long make the component; of a
    kept component only the original pieces at least ``min_piece_mm`` long are returned, so
    a lead label printed just before the trace (letters ~2 mm) does not move its start.
    ``exclude_boxes`` (x0, y0, x1, y1): pieces lying wholly inside one — a calibration pulse
    standing apart from its trace — are not joined to the trace (its foot ends ~1 mm
    before the trace on a 3x4 page and would move the start onto the pulse).
    """
    import cv2

    src = ink.astype(np.uint8)
    if bridge_mm > 0 and exclude_boxes:
        n0, lab0, st0, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        drop = np.zeros(n0, dtype=bool)
        for i in range(1, n0):
            x, y, w, h = st0[i, :4]
            drop[i] = any(bx0 <= x and x + w <= bx1 and by0 <= y and y + h <= by1
                          for bx0, by0, bx1, by1 in exclude_boxes)
        src = np.where(drop[lab0], 0, src).astype(np.uint8)
        ink = src.astype(bool)
    joined = src
    if bridge_mm > 0:
        k = max(1, int(round(bridge_mm * px_per_mm))) | 1    # odd: the closing does not shift the ends
        joined = cv2.morphologyEx(src, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(stats[:, cv2.CC_STAT_WIDTH] >= min_width_mm * px_per_mm):
        if i == 0:
            continue
        x, y, w, h = stats[i, :4]
        if h < min_height_mm * px_per_mm:
            continue
        col = (labels[y:y + h, x:x + w] == i).sum(axis=0)
        keep[i] = col.max() >= min_height_mm * px_per_mm
    out = keep[labels]
    if bridge_mm <= 0:
        return out
    out &= ink.astype(bool)
    _, lab2, st2, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
    big = st2[:, cv2.CC_STAT_WIDTH] >= min_piece_mm * px_per_mm
    big[0] = False
    return big[lab2]


def row_baselines(ink: np.ndarray, px_per_mm: float, *, min_gap_mm: float = 10.0, rel_height: float = 0.25,
                  min_width_mm: float = 20.0, min_height_mm: float = 3.0, bridge_mm: float = 0.0,
                  exclude_boxes: list[tuple[int, int, int, int]] | None = None) -> list[int]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    strokes = long_stroke_mask(ink, px_per_mm, min_width_mm, min_height_mm, bridge_mm=bridge_mm,
                               exclude_boxes=exclude_boxes)
    prof = gaussian_filter1d(strokes.sum(axis=1).astype(np.float64), sigma=max(1.0, 0.8 * px_per_mm))
    if prof.max() <= 0:
        return []
    peaks, _ = find_peaks(prof, height=rel_height * prof.max(), distance=max(1, int(min_gap_mm * px_per_mm)))
    return [int(p) for p in peaks]
