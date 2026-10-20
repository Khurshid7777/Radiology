"""Single-column strips: N leads stacked as full-width rows (TZ §4, D72).

Many bedside machines print a "12L" strip — twelve rows, lead labels at the left
margin, a 1-mV calibration pulse before every few rows, all leads simultaneous
over the whole width. Phone photos of such strips are what reach the Telegram
collector. The 3x4 page digitiser cannot read them (it expects four rows), so
this module reads any ``rows x 1`` layout:

1. ink mask from colour, not brightness alone — the trace is dark and neutral,
   the grid is red, a ballpoint annotation is blue-ish; at phone resolution
   (~4 px/mm) the trace is one pixel wide and fails the page mask's 0.6 ratio;
2. the photo is rectified by ``dewarp_page`` and the mask is warped with the
   same map (``apply_dewarp``);
3. row baselines are found per vertical tile (ink piles up on the flat segments)
   and linked across tiles, so curl and residual tilt follow the paper;
4. each row is straightened along its baseline and read with the page tracker
   (``_trace_rows_tracked``), which follows one trace through crossings;
5. the calibration pulses give mm/mV, the grid pitch gives px/mm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import STANDARD_12_LEADS, Provenance, SignalRepresentation
from cardiolens.layer1_ingestion.digitize import (
    PULSE_AREA_MM,
    PULSE_HEIGHT_MV,
    PULSE_WIDTH_MM,
    _column_runs,
    _draw_grid,
    _draw_trace,
    _fill_gaps,
    _keep_long_components,
    _trace_rows_tracked,
    apply_dewarp,
    dewarp_page,
    estimate_grid_pitch,
    thin_mask,
)

STRIP_12_ROWS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STRIP_INK_RATIO = 0.78        # darker than this fraction of the local brightness = ink (page mask: 0.6)
STRIP_INK_MAX_CHROMA = 45     # ... and at most this max-min channel spread (the red grid is coloured)
STRIP_PEN_MIN_BR = 25         # blue - red at least this: a ballpoint stroke (painted over for the grid fit)
STRIP_ROW_TILE_MM = 32.0      # baselines are found per tile this wide
STRIP_MIN_ROW_GAP_MM = 8.0    # two baselines closer than this are one row
STRIP_ROW_LINK_TOL = 0.35     # of the row pitch: the lattice offset may move this much between neighbouring tiles
STRIP_ROW_MATCH_TOL = 0.08    # of the row pitch: a tile's peak this close to a lattice row is that row's baseline (a label sits ~0.17 away)
STRIP_TILE_PEAK_FRACTION = 0.15
STRIP_PAPER_MIN_BRIGHTNESS = 90
STRIP_PAPER_FRACTION = 0.9
STRIP_BAND_PITCHES = 1.5      # the band read around a baseline, in row pitches each way
STRIP_BOX_PITCHES = 0.75      # the tracker's box, in row pitches each way (its baseline pull is to the own row)
STRIP_TEXT_MARGIN_PITCHES = 0.7   # ink farther than this outside the first/last baseline is header/footer text
STRIP_PULSE_SEARCH_MM = 20.0  # the calibration pulse is looked for within this much of the left edge
STRIP_PULSE_MIN_MM = 5.0      # a pulse top closer than this to the baseline is the trace itself
STRIP_PULSE_MAX_MM = 30.0
STRIP_PULSE_MAX_PITCHES = 0.75   # ... and never this close to the row above (its baseline is a horizontal run too)
STRIP_PULSE_AGREE = 0.15      # pulses within this fraction of their median height agree; two agreeing set the scale
STRIP_FOREIGN_MIN_PITCHES = 0.8   # a flat plateau at least this far from the own baseline is a neighbour's trace
STRIP_FOREIGN_FLAT_PITCHES = 0.12
STRIP_FOREIGN_MIN_COLUMNS = 3
STRIP_NEIGHBOUR_PITCHES = (0.85, 1.3)   # a run of STRIP_NEIGHBOUR_MIN_COLUMNS at this distance sits on the next row
STRIP_NEIGHBOUR_MIN_COLUMNS = 4
STRIP_JUMP_MAX_COLUMNS = 3        # a shorter excursion to a neighbour's baseline between agreeing columns is a bridge
STRIP_FLAT_STROKES = 3.0          # an ink run at most this many stroke widths high is flat trace, not a QRS stroke
STRIP_BASELINE_WEIGHT = 1.0       # tracker pull to the own (straightened) baseline; the page's 0.25 lets the path ride a neighbour
STRIP_LINE_MIN_PITCHES = 0.6      # an ink run this tall in one column, between two short columns, is a grid line
STRIP_LINE_MAX_COLUMNS = 3
STRIP_INK_MAX_RB = 15             # r - b above this is the red grid, not the black trace
STRIP_GRID_FRACTION = 0.3         # of the median red-grid density: a row/column with less is off the paper
STRIP_GLYPH_MAX_MM = 4.0          # printed text: glyph height at most this, width (a word) at most 4x
STRIP_TEXT_MIN_GLYPHS = 5         # ... at least this many on one line
STRIP_TEXT_MIN_SPAN_MM = 40.0     # ... spanning at least this much
STRIP_TEXT_MIN_BASELINE_PITCHES = 0.3   # ... and not on a row baseline
STRIP_BLOCK_MIN_S = 1.0         # a sheet's row block is at least this long (at 25 mm/s) (D126)
STRIP_BLOCK_FIT_MARGIN = 0.15   # ... its rows are fitted over the block without this fraction at each end
STRIP_BLOCK_DROP = 0.55         # ... and ends where its row count falls below this fraction
STRIP_SHIFT_MIN_ROWS = 0.5      # the row lattice moves when that gains this many rows' worth of trace ink (D126)
STRIP_STANDARD_MM_PER_MV = (5.0, 10.0, 20.0)   # the gains recorders print (D128)
STRIP_SCALE_CONFLICT_TOL = 0.05  # pulses implying none of them within this: scale conflict
SCALE_CONFLICT_NOTE = "scale conflict: the grid pitch and the calibration pulses disagree"
STRIP_LABEL_GLYPH_MM = 5.0      # a lead label's letter: a component at most this wide and high (D125)
STRIP_ROW_PITCH_MM = 18.0     # renderer default
STRIP_LABEL_MM = 10.0         # renderer: left margin for labels / pulse


def ink_mask_colour(bgr: np.ndarray, px_per_mm: float = 4.0, ratio: float = STRIP_INK_RATIO) -> np.ndarray:
    """Ink = dark relative to the local paper brightness, unsaturated, not blue pen,
    not a large dark region (the desk around the paper)."""
    import cv2

    b, g, r = cv2.split(bgr.astype(np.int16))
    v = bgr.max(axis=2).astype(np.float32)
    k = 2 * int(4 * px_per_mm) + 1
    local = cv2.blur(v, (k, k)) + 1.0
    # Neutral ink by absolute chroma (max - min of the channels), not HSV saturation:
    # on a dark pixel JPEG chroma noise of a few levels is a huge HSV saturation
    # and would cut the trace to pieces; the red grid's heavy lines keep a
    # chroma of several tens, a ballpoint stroke a clear b > r.
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    ink = (v < ratio * local) & (chroma <= STRIP_INK_MAX_CHROMA)
    pen = ((b - r) >= STRIP_PEN_MIN_BR) & ((b - g) >= STRIP_PEN_MIN_BR // 3)
    pen = cv2.dilate(pen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ink &= ~pen
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.02 * ink.size:
            big[i] = True
    return ink & ~big[labels]


def paint_pen(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Grayscale with blue-pen strokes replaced by the local paper brightness, so
    the grid-texture de-warp and the pitch estimate do not see them."""
    import cv2

    b, g, r = cv2.split(bgr.astype(np.int16))
    pen = ((b - r) >= STRIP_PEN_MIN_BR) & ((b - g) >= STRIP_PEN_MIN_BR // 3)
    pen = cv2.dilate(pen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out = gray.copy()
    out[pen] = cv2.medianBlur(gray, 21)[pen]
    return out


def paper_box(gray: np.ndarray, bgr: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the rows/columns that are mostly paper.

    With the colour image, paper is where the red grid is printed (a wooden
    desk or a shadowed margin is as bright as paper, but has no grid); without
    it, brightness alone."""
    import cv2

    if bgr is not None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue, sat, _v = cv2.split(hsv)
        red = (sat > 50) & ((hue < 15) | (hue > 165))
        rows_p, cols_p = red.mean(axis=1), red.mean(axis=0)
        rows = np.flatnonzero(rows_p > STRIP_GRID_FRACTION * np.median(rows_p[rows_p > 0]) if rows_p.any() else rows_p)
        cols = np.flatnonzero(cols_p > STRIP_GRID_FRACTION * np.median(cols_p[cols_p > 0]) if cols_p.any() else cols_p)
    else:
        paper = gray > STRIP_PAPER_MIN_BRIGHTNESS
        rows = np.flatnonzero(paper.mean(axis=1) > STRIP_PAPER_FRACTION)
        cols = np.flatnonzero(paper.mean(axis=0) > STRIP_PAPER_FRACTION)
    if rows.size < 10 or cols.size < 10:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _tile_peaks(profile: np.ndarray, min_gap: int) -> list[int]:
    sm = np.convolve(profile, np.ones(5) / 5.0, mode="same")
    if sm.max() <= 0:
        return []
    thr = STRIP_TILE_PEAK_FRACTION * sm.max()
    peaks = []
    for i in range(min_gap, len(sm) - min_gap):
        if sm[i] > thr and sm[i] == sm[i - min_gap:i + min_gap + 1].max():
            if peaks and i - peaks[-1] < min_gap:
                continue
            peaks.append(i)
    return peaks


def row_baselines(mask: np.ndarray, px_per_mm: float, n_rows: int,
                  box: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    """Baseline y of every row at every column, (n_rows, width), and the row pitch.

    Per vertical tile the ink projection peaks on the flat segments of each
    trace; the tile with the expected number of evenly spaced peaks anchors the
    rows and the others are linked to it outward, so a curled or tilted strip is
    followed instead of smeared."""
    x0, y0, x1, y1 = box
    sub = mask[y0:y1, x0:x1]
    tile = max(16, int(STRIP_ROW_TILE_MM * px_per_mm))
    step = max(8, tile // 2)
    min_gap = max(3, int(STRIP_MIN_ROW_GAP_MM * px_per_mm))
    centres, peaks = [], []
    for x in range(0, max(1, sub.shape[1] - tile + 1), step):
        prof = sub[:, x:x + tile].sum(axis=1).astype(float)
        centres.append(x0 + x + tile // 2)
        peaks.append([y0 + p for p in _tile_peaks(prof, min_gap)])
    counts = [len(p) for p in peaks]
    if not counts or max(counts) < n_rows:
        raise ValueError(f"strip: expected {n_rows} rows, found at most {max(counts) if counts else 0}")
    # Anchor: the tile with exactly n_rows peaks and the most regular spacing.
    best, best_cv = None, np.inf
    for i, p in enumerate(peaks):
        if len(p) != n_rows:
            continue
        d = np.diff(p)
        if n_rows == 1:          # a single trace (a rhythm strip, D126): any tile with one peak
            cv = 0.0
        else:
            cv = float(d.std() / d.mean()) if d.mean() > 0 else np.inf
        if cv < best_cv:
            best, best_cv = i, cv
    if best is None:
        raise ValueError(f"strip: no tile shows {n_rows} evenly spaced rows")
    pitch = (float(np.median(np.diff(peaks[best]))) if n_rows > 1
             else min(STRIP_ROW_PITCH_MM * px_per_mm, (y1 - y0) / 2.0))
    # Rows are a lattice of known pitch: per tile, choose the lattice offset that
    # matches most peaks (a T-wave top or a label makes a stray peak, a shadow
    # loses one — neither moves the lattice), staying close to the neighbouring
    # tile's offset, then take each row from its matched peak or the lattice.
    rows = np.full((n_rows, len(centres)), np.nan)
    rows[:, best] = peaks[best]
    match_tol = STRIP_ROW_MATCH_TOL * pitch
    offsets = np.full(len(centres), np.nan)
    offsets[best] = float(np.mean(np.asarray(peaks[best]) - np.arange(n_rows) * pitch))
    for order in (range(best - 1, -1, -1), range(best + 1, len(centres))):
        prev_off = offsets[best]
        for i in order:
            cand = np.array(peaks[i], dtype=float)
            lattice = np.arange(n_rows) * pitch
            best_off, best_score = prev_off, -1.0
            for p in cand:
                for r in range(n_rows):
                    off = p - r * pitch
                    if abs(off - prev_off) > STRIP_ROW_LINK_TOL * pitch:
                        continue
                    dist = np.abs(cand[:, None] - (lattice + off)[None, :]).min(axis=1)
                    score = float((dist <= match_tol).sum()) - 0.1 * abs(off - prev_off) / pitch
                    if score > best_score:
                        best_off, best_score = off, score
            offsets[i] = best_off
            for r in range(n_rows):
                target = best_off + r * pitch
                if cand.size:
                    j = int(np.argmin(np.abs(cand - target)))
                    rows[r, i] = cand[j] if abs(cand[j] - target) <= match_tol else target
                else:
                    rows[r, i] = target
            prev_off = best_off
    # Keep the anchor tile on the same rule (a stray peak there is as likely).
    lattice0 = offsets[best] + np.arange(n_rows) * pitch
    anchor = np.asarray(peaks[best], dtype=float)
    for r in range(n_rows):
        j = int(np.argmin(np.abs(anchor - lattice0[r])))
        rows[r, best] = anchor[j] if abs(anchor[j] - lattice0[r]) <= match_tol else lattice0[r]
    width = mask.shape[1]
    xs = np.arange(width)
    base = np.zeros((n_rows, width))
    cx = np.array(centres, dtype=float)
    for r in range(n_rows):
        ok = np.isfinite(rows[r])
        if ok.sum() >= 2:
            base[r] = np.interp(xs, cx[ok], rows[r, ok])
        else:
            base[r] = rows[r, best]
    return base, pitch


def _find_pulse(band: np.ndarray, x_left: int, half: int, px_per_mm: float,
                max_height_px: float | None = None) -> tuple[float, int] | None:
    """(height px, right edge x) of the 1-mV rectangle in the first
    STRIP_PULSE_SEARCH_MM of a straightened row.

    The page detector measures the rectangle's vertical edges, which a 4-px/mm
    photo breaks into dots; here the band is already straightened on the
    baseline, so the height is the distance from the top edge — the highest
    horizontal ink run at least 0.7 x PULSE_WIDTH_MM wide, between
    STRIP_PULSE_MIN_MM and STRIP_PULSE_MAX_MM above the baseline, with ink
    continuing downward at both ends — to the baseline."""
    import cv2

    span = int(STRIP_PULSE_SEARCH_MM * px_per_mm)
    sub = band[:, x_left:x_left + span].astype(np.uint8)
    if sub.size == 0:
        return None
    # a 1-px printed line breaks into dots after the warp: close 1-px gaps along x
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)).astype(bool)
    min_w = int(0.7 * PULSE_WIDTH_MM * px_per_mm)
    top = STRIP_PULSE_MAX_MM * px_per_mm
    if max_height_px is not None:          # never as high as the row above (its baseline is a long run too)
        top = min(top, max_height_px)
    lo, hi = half - int(top), half - int(STRIP_PULSE_MIN_MM * px_per_mm)
    for y in range(max(0, lo), max(0, hi)):
        cols = np.flatnonzero(sub[y])
        if cols.size < min_w:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, cols.size - 1]
        k = int(np.argmax(ends - starts))
        a, b = int(cols[starts[k]]), int(cols[ends[k]])
        if b - a + 1 < min_w or b - a + 1 > 2 * PULSE_WIDTH_MM * px_per_mm:
            continue
        # vertical edges: ink below the top run near both ends — a third of the
        # height on one side, a few pixels on the other (one edge is often dotted)
        need = max(2, int(0.3 * (half - y)))
        left = int(sub[y:y + 3 * need, max(0, a - 1):a + 2].any(axis=1).sum())
        right = int(sub[y:y + 3 * need, b - 1:b + 2].any(axis=1).sum())
        if max(left, right) >= need:       # the other edge is often a dotted line at phone resolution
            return float(half - y), x_left + b + 1
    return None


def _run_heights(band: np.ndarray, trace: np.ndarray) -> tuple[np.ndarray, float]:
    """Height of the ink run the trace sits in, per column, and the stroke width
    (median run height): a flat segment is one stroke high, a QRS stroke many."""
    n = len(trace)
    heights = np.zeros(n)
    for i in range(n):
        if not np.isfinite(trace[i]):
            continue
        for a, b in _column_runs(band[:, i]):
            if a - 1 <= trace[i] <= b + 1:
                heights[i] = b - a + 1
                break
    stroke = float(np.median(heights[heights > 0])) if (heights > 0).any() else 2.0
    return heights, stroke


def _drop_short_jumps(trace: np.ndarray, half: int, pitch: float, heights: np.ndarray, stroke: float) -> np.ndarray:
    """NaN out a run of at most STRIP_JUMP_MAX_COLUMNS columns that sits about one
    row pitch away from the own baseline, on flat ink, while the columns on both
    sides agree near it: the tracker bridged a gap in the own trace with the
    neighbour's flat trace. A real R wave of that height is a tall stroke in
    those columns, not one stroke high."""
    out = trace.copy()
    n = len(trace)
    d = trace - half
    i = 0
    while i < n:
        if not np.isfinite(out[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(out[j + 1]) and abs(d[j + 1] - d[i]) <= 0.15 * pitch:
            j += 1
        width = j - i + 1
        before = d[i - 1] if i >= 1 and np.isfinite(out[i - 1]) else None
        after = d[j + 1] if j + 1 < n and np.isfinite(out[j + 1]) else None
        if (width <= STRIP_JUMP_MAX_COLUMNS and before is not None and after is not None
                and abs(before - after) <= 0.15 * pitch
                and abs(d[i] - before) >= 0.5 * pitch and abs(d[j] - after) >= 0.5 * pitch
                and STRIP_NEIGHBOUR_PITCHES[0] <= abs(np.median(d[i:j + 1])) / pitch <= STRIP_NEIGHBOUR_PITCHES[1]
                and np.all(heights[i:j + 1] <= STRIP_FLAT_STROKES * stroke)):
            out[i:j + 1] = np.nan
        i = j + 1
    return out


def _erase_text_lines(mask: np.ndarray, px_per_mm: float, base: np.ndarray, pitch: float) -> np.ndarray:
    """Erase printed header/footer lines: many small glyph components aligned on
    one line, away from every row baseline (lead labels are isolated; a broken
    faint trace lies on its baseline and is kept)."""
    import cv2

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    max_h, max_w = STRIP_GLYPH_MAX_MM * px_per_mm, 4 * STRIP_GLYPH_MAX_MM * px_per_mm
    small = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_HEIGHT] <= max_h and stats[i, cv2.CC_STAT_WIDTH] <= max_w]
    if len(small) < STRIP_TEXT_MIN_GLYPHS:
        return mask
    bin_px = max(2.0, 2.0 * px_per_mm)
    bins: dict[int, list[int]] = {}
    for i in small:
        bins.setdefault(int(cents[i][1] // bin_px), []).append(i)
    erase = np.zeros(n, dtype=bool)
    for key, members in bins.items():
        if len(members) < STRIP_TEXT_MIN_GLYPHS:
            continue
        xs = cents[members, 0]
        if xs.max() - xs.min() < STRIP_TEXT_MIN_SPAN_MM * px_per_mm:
            continue
        y_line = (key + 0.5) * bin_px
        x_mid = int(np.clip(np.median(xs), 0, base.shape[1] - 1))
        if np.min(np.abs(base[:, x_mid] - y_line)) < STRIP_TEXT_MIN_BASELINE_PITCHES * pitch:
            continue
        erase[members] = True
    return mask & ~erase[labels]


def _drop_foreign_plateaus(trace: np.ndarray, half: int, pitch: float,
                           heights: np.ndarray, stroke: float) -> np.ndarray:
    """NaN out runs where the tracker sat on a neighbouring row's flat trace (a
    plateau STRIP_FOREIGN_MIN_PITCHES or more from the own baseline, on ink one
    stroke high) or on the band edge; the caller interpolates them like any
    other gap. A tall R wave reaching the same height is a tall stroke and stays."""
    out = trace.copy()
    d = np.abs(trace - half)
    edge = d >= STRIP_BAND_PITCHES * pitch - 2
    out[edge] = np.nan
    far = d >= STRIP_FOREIGN_MIN_PITCHES * pitch
    n = len(trace)
    i = 0
    while i < n:
        if not (far[i] and np.isfinite(out[i])):
            i += 1
            continue
        j = i
        while j + 1 < n and far[j + 1] and np.isfinite(out[j + 1]):
            j += 1
        seg = trace[i:j + 1]
        width = j - i + 1
        flat_ink = np.all(heights[i:j + 1] <= STRIP_FLAT_STROKES * stroke)
        flat = width >= STRIP_FOREIGN_MIN_COLUMNS and np.ptp(seg) <= STRIP_FOREIGN_FLAT_PITCHES * pitch
        med = float(np.median(np.abs(seg - half))) / pitch
        on_neighbour = (width >= STRIP_NEIGHBOUR_MIN_COLUMNS
                        and STRIP_NEIGHBOUR_PITCHES[0] <= med <= STRIP_NEIGHBOUR_PITCHES[1])
        if flat_ink and (flat or on_neighbour):
            out[i:j + 1] = np.nan
        i = j + 1
    return out


def _drop_line_crossings(trace: np.ndarray, pitch: float, heights: np.ndarray, stroke: float) -> np.ndarray:
    """NaN out columns where the tracker read the middle of a vertical grid line
    crossing a flat trace: an ink run at least STRIP_LINE_MIN_PITCHES tall, at most
    STRIP_LINE_MAX_COLUMNS wide, between two columns whose own runs are short and
    agree with each other. A steep QRS stroke is tall in its neighbours too."""
    n = len(trace)
    tall = heights >= STRIP_LINE_MIN_PITCHES * pitch
    short = (heights > 0) & (heights <= STRIP_FLAT_STROKES * stroke)
    out = trace.copy()
    i = 0
    while i < n:
        if not tall[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and tall[j + 1]:
            j += 1
        if (j - i + 1 <= STRIP_LINE_MAX_COLUMNS and i >= 1 and j + 1 < n and short[i - 1] and short[j + 1]
                and abs(trace[i - 1] - trace[j + 1]) <= 0.1 * pitch):
            out[i:j + 1] = np.nan
        i = j + 1
    return out


def _drop_glyphs(band: np.ndarray, px_per_mm: float) -> np.ndarray:
    """The band without components at most STRIP_LABEL_GLYPH_MM in both directions."""
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    lim = STRIP_LABEL_GLYPH_MM * px_per_mm
    small = np.zeros(n, dtype=bool)
    small[1:] = (st[1:, cv2.CC_STAT_WIDTH] <= lim) & (st[1:, cv2.CC_STAT_HEIGHT] <= lim)
    return band & ~small[lab]


def _shift_off_text(mask: np.ndarray, base: np.ndarray, pitch: float, px_per_mm: float,
                    box: tuple[int, int, int, int]) -> np.ndarray:
    """The row lattice moved by -1, 0 or +1 pitch, whichever puts the most trace ink near the
    baselines. A header line above the first row (a sheet's title) can take the lattice's
    first place and push the last row off it (D126); its glyphs do not count."""
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    lim = STRIP_LABEL_GLYPH_MM * px_per_mm
    small = np.zeros(n, dtype=bool)
    small[1:] = (st[1:, cv2.CC_STAT_WIDTH] <= lim) & (st[1:, cv2.CC_STAT_HEIGHT] <= lim)
    trace = mask & ~small[lab]
    ys = np.arange(mask.shape[0])[:, None]

    def score(b: np.ndarray) -> float:
        if b.min() < box[1] or b.max() > box[3] - 1:
            return -1.0
        return float(sum((trace & (np.abs(ys - row[None, :]) <= 0.3 * pitch)).any(axis=0).mean() for row in b))

    shifts = {k: score(base + k * pitch) for k in (-1, 0, 1)}
    k = max(shifts, key=shifts.get)
    # Move only on a clear gain: one row's worth of trace ink.
    if shifts[k] - shifts[0] < STRIP_SHIFT_MIN_ROWS * (shifts[0] / len(base)):
        k = 0
    _shift_off_text.last_shift = k  # type: ignore[attr-defined]
    return base + k * pitch


def straighten(mask: np.ndarray, baseline: np.ndarray, half: int) -> np.ndarray:
    """Band of height ``2*half`` whose middle row is the row's baseline at every column."""
    h, w = mask.shape
    band = np.zeros((2 * half, w), dtype=bool)
    yc = np.round(baseline).astype(int)
    for x in range(w):
        a, b = yc[x] - half, yc[x] + half
        sa, sb = max(0, a), min(h, b)
        if sb > sa:
            band[sa - a: sa - a + (sb - sa), x] = mask[sa:sb, x]
    return band


def digitize_strip(
    path: str | Path | np.ndarray, *, rows: int = 12, lead_order: list[str] | None = None,
    px_per_mm: float | None = None, mm_per_s: float = 25.0, mm_per_mv: float | None = None,
    dewarp: bool = True, ink_ratio: float = STRIP_INK_RATIO, block_fraction: float | None = None,
) -> dict[str, np.ndarray]:
    """``{lead: trace_uv}`` at ``px_per_s`` Hz for a rows x 1 strip photo;
    ``digitize_strip.last_calibration`` / ``last_coverage`` hold the provenance."""
    import cv2

    lead_order = lead_order or STRIP_12_ROWS
    if len(lead_order) != rows:
        raise ValueError("lead_order must name one lead per row")
    bgr = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clean = paint_pen(gray, bgr)
    est0 = estimate_grid_pitch(clean)
    mask = ink_mask_colour(bgr, est0 or 4.0, ink_ratio)
    if dewarp:
        clean = dewarp_page(clean)
        mask = apply_dewarp(mask, nearest=True)
        bgr = apply_dewarp(bgr, border=0)
    calib: dict = {"px_per_mm_given": px_per_mm, "mm_per_mv_given": mm_per_mv, "layout": f"{rows}x1"}
    if px_per_mm is None:
        est = estimate_grid_pitch(clean)
        px_per_mm = est if est is not None else (est0 or 4.0)
        calib["px_per_mm_estimated"] = round(float(px_per_mm), 2)
    box = paper_box(clean)
    calib["paper_box"] = [int(v) for v in box]
    outside = np.ones_like(mask)
    outside[box[1]:box[3], box[0]:box[2]] = False
    mask[outside] = False
    mask = _keep_long_components(mask, int(2 * px_per_mm))
    base, pitch = row_baselines(mask, px_per_mm, rows, box)
    if block_fraction is not None:
        base = _shift_off_text(mask, base, pitch, px_per_mm, box)
        calib["row_shift"] = getattr(_shift_off_text, "last_shift", 0)
    calib["row_pitch_mm"] = round(pitch / px_per_mm, 2)
    # Header and footer text: lines of glyphs off every baseline, and anything far
    # outside the first/last baseline.
    mask = _erase_text_lines(mask, px_per_mm, base, pitch)
    ys = np.arange(mask.shape[0])[:, None]
    margin = STRIP_TEXT_MARGIN_PITCHES * pitch
    mask &= (ys >= base[0][None, :] - margin) & (ys <= base[-1][None, :] + margin)
    # The band around a baseline spans STRIP_BAND_PITCHES row pitches each way: a
    # 2-mV R wave in V3 leaves its 15-mm row. The tracker's box is narrower (its
    # baseline pull is to the own row) and it extends the search by half a box.
    half = max(4, int(round(STRIP_BAND_PITCHES * pitch)))
    y0, y1 = half - int(round(STRIP_BOX_PITCHES * pitch)), half + int(round(STRIP_BOX_PITCHES * pitch))
    # The traces' own extent: columns where the middle rows carry ink close to
    # their baselines (the desk, a shadowed margin and the warp border have none).
    mid = range(rows // 3, rows - rows // 3)
    near = np.zeros(mask.shape[1], dtype=int)
    ys_all = np.arange(mask.shape[0])[:, None]
    for r in mid:
        near += (mask & (np.abs(ys_all - base[r][None, :]) <= 0.3 * pitch)).any(axis=0)
    cols_ink = np.flatnonzero(near >= max(1, len(mid) // 2))
    if block_fraction is not None:
        # A sheet with the rows in one block and text beside it (Cardipia, D126): the
        # block is where at least this fraction of all rows carries ink near its
        # baseline over a sliding 5 mm (text sits beside a few rows only).
        every = np.zeros(mask.shape[1])
        for r in range(rows):
            every += (mask & (np.abs(ys_all - base[r][None, :]) <= 0.3 * pitch)).any(axis=0)
        k = max(1, int(5 * px_per_mm))
        every = np.convolve(np.minimum(every, rows), np.ones(k) / k, mode="same")
        cols_ink = np.flatnonzero(every >= block_fraction * rows)
        # The block ends where the count steps down (text beside it reaches a few rows,
        # a rhythm strip one): the first column where the mean over the next
        # STRIP_BLOCK_MIN_S falls below STRIP_BLOCK_DROP of the mean so far.
        if cols_ink.size:
            a, b = int(cols_ink[0]), int(box[2])
            seg = every[a:b]
            csum = np.concatenate([[0.0], np.cumsum(seg)])
            n = len(seg)
            lo = min(n - 1, int(STRIP_BLOCK_MIN_S * 25.0 * px_per_mm))
            xs_split = np.arange(max(1, lo), n - lo)
            if xs_split.size:
                left = csum[xs_split] / xs_split
                right = (csum[xs_split + lo] - csum[xs_split]) / lo
                drop = np.flatnonzero(right < STRIP_BLOCK_DROP * left)
                if drop.size:
                    # The window ahead reaches past the edge before the column does:
                    # the edge is the last column within that window still at half
                    # the block's mean count.
                    x_d = int(xs_split[drop[0]])
                    win = seg[x_d:x_d + lo]
                    keep = np.flatnonzero(win >= 0.5 * left[drop[0]])
                    cut = a + x_d + (int(keep[-1]) + 1 if keep.size else 0)
                    cols_ink = cols_ink[cols_ink < cut]
    x_right = min(box[2], int(cols_ink[-1]) + int(px_per_mm)) if cols_ink.size else box[2] - int(px_per_mm)
    if block_fraction is not None and cols_ink.size:
        # On a sheet the rows are straight once de-warped, but the tiles over the label
        # margin and the text beside the block pull the linked baselines off them:
        # each row becomes the line fitted over the middle of the block.
        a, b = int(cols_ink[0]), x_right
        a, b = a + int(STRIP_BLOCK_FIT_MARGIN * (b - a)), b - int(STRIP_BLOCK_FIT_MARGIN * (b - a))
        xs = np.arange(mask.shape[1])
        for r in range(rows):
            k, c = np.polyfit(xs[a:b], base[r, a:b], 1)
            base[r] = k * xs + c
    # Calibration pulses (searched from the paper's left edge, before the traces
    # begin): 1 mV -> px; each row's trace starts where its own ink begins after
    # its pulse (rows without a pulse start a little earlier on some printers),
    # and the common time origin is where most rows have started.
    pulse_heights, row_starts = [], []
    bands = [straighten(mask, base[r], half) for r in range(rows)]
    near_band = slice(half - int(0.3 * pitch), half + int(0.3 * pitch) + 1)
    for band in bands:
        pulse = _find_pulse(band, box[0] + int(px_per_mm), half, px_per_mm,
                            max_height_px=STRIP_PULSE_MAX_PITCHES * pitch)
        from_x = box[0] + int(px_per_mm)
        if pulse is not None:
            pulse_heights.append(pulse[0])
            from_x = pulse[1] + int(0.5 * px_per_mm)
        ink_cols = np.flatnonzero(band[near_band, from_x:].any(axis=0))
        if ink_cols.size:
            row_starts.append(from_x + int(ink_cols[0]))
    # The scale is taken from the pulses only when at least two agree: one
    # rectangle-like run (a label, a boxed annotation) must not rescale the record.
    agreed = []
    if pulse_heights:
        med = float(np.median(pulse_heights))
        agreed = [h for h in pulse_heights if abs(h - med) <= STRIP_PULSE_AGREE * med]
    calib["pulses_found"] = len(pulse_heights)
    if len(agreed) >= 2:
        if mm_per_mv is None:
            mm_per_mv = float(np.median(agreed)) / px_per_mm / PULSE_HEIGHT_MV
        calib["mm_per_mv_from_pulse"] = round(float(np.median(agreed)) / px_per_mm, 2)
        implied = float(np.median(agreed)) / px_per_mm / PULSE_HEIGHT_MV
        if (calib["px_per_mm_given"] is None and calib["mm_per_mv_given"] is None
                and min(abs(implied / g - 1.0) for g in STRIP_STANDARD_MM_PER_MV) > STRIP_SCALE_CONFLICT_TOL):
            # No recorder prints 5.5 mm/mV: the grid pitch and the pulses disagree. At
            # ~4 px/mm a blur merges the 1-mm lines and the grid estimate lands between
            # harmonics (6.88 for 4.0): a 60/min record read at 103/min that Einthoven
            # does not notice. Which one is right is not known, so it is flagged (D128).
            calib["scale_conflict"] = True
    pulse_heights = agreed if len(agreed) >= 2 else []
    if row_starts:
        x_left = int(np.percentile(row_starts, 75))
    elif cols_ink.size:
        x_left = max(box[0], int(cols_ink[0]))
    else:
        x_left = box[0] + int((PULSE_AREA_MM + PULSE_WIDTH_MM) * px_per_mm)
    mm_per_mv = mm_per_mv or 10.0
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    calib.update(px_per_mm=round(float(px_per_mm), 2), mm_per_mv=round(float(mm_per_mv), 2),
                 pulse_found=bool(pulse_heights), pulses=len(pulse_heights), trace_x=(int(x_left), int(x_right)))
    out: dict[str, np.ndarray] = {}
    coverages = []
    # Ink close to each baseline per column (x_left..x_right): a sequential layout's
    # group boundaries are where every row has none (D125).
    # Lead labels printed on the baseline are glyph-sized components and do not count.
    digitize_strip.last_near_ink = np.array(  # type: ignore[attr-defined]
        [_drop_glyphs(band, px_per_mm)[near_band, x_left:x_right].any(axis=0) for band in bands])
    for lead, band in zip(lead_order, bands, strict=True):
        thin = thin_mask(band)
        raw = _trace_rows_tracked(band, thin, x_left, x_right, y0, y1, baseline_weight=STRIP_BASELINE_WEIGHT)
        heights, stroke = _run_heights(band[:, x_left:x_right], raw)
        raw = _drop_foreign_plateaus(raw, half, pitch, heights, stroke)
        raw = _drop_short_jumps(raw, half, pitch, heights, stroke)
        raw = _drop_line_crossings(raw, pitch, heights, stroke)
        trace, cov = _fill_gaps(raw)
        out[lead] = ((float(np.median(trace)) - trace) * uv_per_px).astype(np.float32)
        coverages.append(cov)
    digitize_strip.last_coverage = float(np.mean(coverages))  # type: ignore[attr-defined]
    digitize_strip.last_calibration = calib  # type: ignore[attr-defined]
    # The rectified photo and the rows, for readers of what lies beside them (D126).
    digitize_strip.last_geometry = {"bgr": bgr, "base": base, "pitch": pitch, "box": box,  # type: ignore[attr-defined]
                                    "trace_x": (int(x_left), int(x_right))}
    return out


def strip_to_signal(traces: dict[str, np.ndarray], *, px_per_mm: float | None = None,
                    mm_per_s: float = 25.0, coverage: float | None = None,
                    calibration: dict | None = None) -> SignalRepresentation:
    """12-lead SignalRepresentation with every lead valid over the whole strip."""
    calibration = calibration or getattr(digitize_strip, "last_calibration", None) or {}
    px = px_per_mm if px_per_mm is not None else calibration.get("px_per_mm", 4.0)
    fs = float(px * mm_per_s)
    n = min(len(t) for t in traces.values())
    signal = np.zeros((12, n), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    for lead, t in traces.items():
        if lead in STANDARD_12_LEADS:
            signal[STANDARD_12_LEADS.index(lead)] = t[:n]
            windows[lead] = (0.0, n / fs)
    cov = coverage if coverage is not None else getattr(digitize_strip, "last_coverage", None)
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS),
        duration_s=float(n / fs), source="image",
        provenance=Provenance(digitized=True, confidence=None if cov is None else round(cov, 3),
                              lead_windows=windows,
                              notes=[f"{calibration.get('layout', '12x1')} strip: all leads simultaneous"]
                                    + ([f"calibration: {calibration}"] if calibration else [])
                                    + ([SCALE_CONFLICT_NOTE] if calibration.get("scale_conflict") else [])),
    )


def digitize_strip_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    traces = digitize_strip(path, **kw)
    return strip_to_signal(traces, mm_per_s=kw.get("mm_per_s", 25.0),
                           coverage=getattr(digitize_strip, "last_coverage", None),
                           calibration=getattr(digitize_strip, "last_calibration", None))


def count_rows(path: str | Path | np.ndarray) -> int:
    """Modal number of trace rows across the strip's tiles (12 for a 12L strip,
    4 for a 3x4 page with a rhythm strip) — the layout switch for ``digitize_auto``."""
    import cv2

    bgr = path if isinstance(path, np.ndarray) else cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clean = paint_pen(gray, bgr)
    ppm = estimate_grid_pitch(clean) or 4.0
    mask = _keep_long_components(ink_mask_colour(bgr, ppm), int(2 * ppm))
    x0, y0, x1, y1 = paper_box(clean)
    sub = mask[y0:y1, x0:x1]
    tile = max(16, int(STRIP_ROW_TILE_MM * ppm))
    min_gap = max(3, int(STRIP_MIN_ROW_GAP_MM * ppm))
    counts = [len(_tile_peaks(sub[:, x:x + tile].sum(axis=1).astype(float), min_gap))
              for x in range(0, max(1, sub.shape[1] - tile + 1), max(8, tile // 2))]
    return int(np.bincount(counts).argmax()) if counts else 0


def digitize_auto_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """Pick the layout: a device screen photo -> screen reader (D120); a narrow strip of
    paper -> three-channel reader (D125); ~12 rows -> a report sheet (rows beside text
    with a rhythm strip, D126), then a strip; otherwise the 3x4 page digitiser; the
    others are the fallbacks in that order."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import digitize_page_to_signal
    from cardiolens.layer1_ingestion.screen import digitize_screen_to_signal, is_device_screen
    from cardiolens.layer1_ingestion.sequential import (
        digitize_sequential_to_signal,
        is_sequential_strip,
    )
    from cardiolens.layer1_ingestion.sheet import digitize_sheet_to_signal

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None and is_device_screen(bgr):     # a device's own display (D120)
        try:
            return digitize_screen_to_signal(path)
        except ValueError:
            pass
    if bgr is not None and is_sequential_strip(bgr):  # three rows, lead groups in turn (D125)
        try:
            return digitize_sequential_to_signal(path)
        except ValueError:
            pass
    rows = count_rows(path)
    if bgr is not None and bgr.shape[0] > bgr.shape[1]:
        # A portrait photo of a landscape sheet (D126): its rows run across the turned frame.
        rows = max(rows, count_rows(np.ascontiguousarray(np.rot90(bgr))))
    order = ([digitize_sheet_to_signal, digitize_strip_to_signal, digitize_page_to_signal] if rows >= 8 else
             [digitize_page_to_signal, digitize_strip_to_signal, digitize_sheet_to_signal])
    first_err = None
    for reader in order:
        try:
            return reader(path) if reader is digitize_sheet_to_signal else reader(path, **kw)
        except ValueError as err:
            first_err = first_err or err
    raise first_err


def render_ecg_strip(sig: SignalRepresentation, path: str | Path, *, px_per_mm: int = 6,
                     mm_per_s: float = 25.0, mm_per_mv: float = 10.0, row_pitch_mm: float = STRIP_ROW_PITCH_MM,
                     pulse_every: int = 3, lead_order: list[str] | None = None) -> str:
    """Render a 12 x 1 strip (labels at the left, a 1-mV pulse before every
    ``pulse_every``-th row) — the synthetic counterpart of a bedside 12L printout."""
    import cv2

    lead_order = lead_order or STRIP_12_ROWS
    px_per_s = px_per_mm * mm_per_s
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    margin = int(STRIP_LABEL_MM * px_per_mm)
    pulse_w = int(PULSE_WIDTH_MM * px_per_mm)
    pitch = int(row_pitch_mm * px_per_mm)
    width = margin + pulse_w + int(sig.duration_s * px_per_s) + margin
    height = pitch * (len(lead_order) + 1)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
    x_trace = margin + pulse_w + int(1 * px_per_mm)
    for r, lead in enumerate(lead_order):
        yc = pitch * (r + 1)
        cv2.putText(img, lead, (int(2 * px_per_mm), yc - int(3 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.08 * px_per_mm, (20, 20, 20), 1, cv2.LINE_AA)
        if pulse_every and r % pulse_every == 0:
            top = yc - int(PULSE_HEIGHT_MV * 1000 * px_per_uv)
            pts = np.array([[margin - pulse_w // 2, yc], [margin - pulse_w // 2, top], [margin + pulse_w // 2, top],
                            [margin + pulse_w // 2, yc]], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 20), thickness=2)
        if lead in sig.lead_names:
            _draw_trace(img, sig.signal[sig.lead_names.index(lead)], sig.sampling_rate_hz, x_trace, yc,
                        px_per_s, px_per_uv)
    cv2.imwrite(str(path), img)
    return str(path)
