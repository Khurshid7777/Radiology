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
                     min_height_mm: float = 3.0) -> np.ndarray:
    """Components at least ``min_width_mm`` long with a vertical stroke of at least
    ``min_height_mm`` in some column: a trace has complexes; a paper edge, a shadow
    border or a ruled line is long and, even when tilted, thin in every column."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(stats[:, cv2.CC_STAT_WIDTH] >= min_width_mm * px_per_mm):
        if i == 0:
            continue
        x, y, w, h = stats[i, :4]
        if h < min_height_mm * px_per_mm:
            continue
        col = (labels[y:y + h, x:x + w] == i).sum(axis=0)
        keep[i] = col.max() >= min_height_mm * px_per_mm
    return keep[labels]


def row_baselines(ink: np.ndarray, px_per_mm: float, *, min_gap_mm: float = 10.0, rel_height: float = 0.25,
                  min_width_mm: float = 20.0) -> list[int]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    strokes = long_stroke_mask(ink, px_per_mm, min_width_mm)
    prof = gaussian_filter1d(strokes.sum(axis=1).astype(np.float64), sigma=max(1.0, 0.8 * px_per_mm))
    if prof.max() <= 0:
        return []
    peaks, _ = find_peaks(prof, height=rel_height * prof.max(), distance=max(1, int(min_gap_mm * px_per_mm)))
    return [int(p) for p in peaks]
