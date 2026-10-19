"""Panels of a page from the layout registry: which lead is where, in pixels and seconds (TZ §5.6).

* **Rows**: the layout needs ``n`` trace rows; the detected baselines may include a
  stray one (a fold, a shadow edge). The subset of ``n`` baselines whose spacing best
  matches the registry pitch (or the median detected pitch when the printer uses
  another) is taken.
* **Time origin** ``x0``: where the traces start — after the calibration pulse of
  each row, the first trace ink near the baseline; the earliest start that another
  row confirms (a row starting off its baseline is found late).
* **Columns**: from the *calibrated* paper speed — column ``c`` of a row spans
  ``[x0 + c * panel_s * speed * ppm, ...)``, so a page is never stretched to the ink
  extent (the legacy reader did, and a desk edge or a shadow then shifted every
  column — M1, SD-019).
* The ink extent is still measured: its length in mm over the record length is an
  independent paper-speed estimate (``implied_speed``), offered as a candidate.
* **Strips** (``panel_gap_mm`` > 0, SD-059): the groups are separated by blank paper, so
  each column's start is read from the page — the first trace ink after a blank stretch
  of at least half the nominal gap — and the speed estimate is the median group width
  over ``panel_s``. Fewer groups than the layout has means the photo does not show the
  whole strip (``column_x0`` stays None).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise

import numpy as np

from ecgscan.layout.registry import LayoutSpec


@dataclass
class Panel:
    lead: str
    kind: str                  # panel | rhythm
    row: int
    t0_s: float
    t1_s: float
    x0: int
    x1: int
    baseline: float


@dataclass
class PageFrame:
    baselines: list[float]
    x0: float
    ink_x1: float | None
    implied_speed_mm_s: float | None
    pitch_px: float
    column_x0: list[float] | None = None     # strips: each group's start read from the page
    groups_found: int | None = None          # strips: groups of trace ink between blank gaps


def pick_rows(baselines: list[int], n: int, pitch_px: float | None) -> list[int] | None:
    """The ``n`` baselines (in order) whose consecutive gaps vary least (and match the pitch)."""
    b = sorted(baselines)
    if len(b) < n:
        b = fill_gaps(b, pitch_px)
    if len(b) < n:
        return None
    if len(b) == n:
        return b
    best, best_cost = None, np.inf
    combos = combinations(b, n) if len(b) <= 18 else [tuple(b[i:i + n]) for i in range(len(b) - n + 1)]
    for c in combos:
        d = np.diff(c)
        ref = pitch_px if pitch_px else float(np.median(d))
        cost = float(np.mean(np.abs(d - ref)) / ref) + float(np.std(d) / max(np.mean(d), 1.0))
        if cost < best_cost:
            best, best_cost = list(c), cost
    return best


def fill_gaps(baselines: list[int], pitch_px: float | None, tol: float = 0.15) -> list[int]:
    """Insert baselines into gaps that are a whole number (>1) of row pitches: a row whose
    flat segments are short (a low-amplitude or very busy lead) can miss the projection peak."""
    b = sorted(baselines)
    if len(b) < 2:
        return b
    d = np.diff(b)
    p = float(np.median(d))
    if pitch_px and abs(p - pitch_px) > 0.25 * pitch_px:
        p = float(pitch_px) if np.min(np.abs(d - pitch_px)) < tol * pitch_px else p
    out = [b[0]]
    for prev, nxt in pairwise(b):
        k = int(round((nxt - prev) / p))
        if k > 1 and abs((nxt - prev) - k * p) <= tol * p:
            out += [int(round(prev + j * (nxt - prev) / k)) for j in range(1, k)]
        out.append(nxt)
    return out


def trace_start(strokes: np.ndarray, baselines: list[float], px_per_mm: float, pulse_right: list[int],
                band_mm: float = 2.5, agree_mm: float = 1.5) -> tuple[float, float | None]:
    """(x0, ink right end): first/last long-stroke ink within ``band_mm`` of each baseline
    (after any calibration pulse); the start that rows agree on, the median end."""
    h, w = strokes.shape
    starts, ends = [], []
    band = int(round(band_mm * px_per_mm))
    for yb in baselines:
        y0, y1 = max(0, int(yb) - band), min(h, int(yb) + band + 1)
        cols = np.flatnonzero(strokes[y0:y1].any(axis=0))
        after = [p for p in pulse_right if p < w * 0.3]
        if after:
            cols = cols[cols > max(after) + int(0.5 * px_per_mm)]
        if cols.size:
            starts.append(cols[0])
            ends.append(cols[-1])
    if not starts:
        return 0.0, None
    return _common_start(starts, agree_mm * px_per_mm), float(np.median(ends)) if ends else None


def _common_start(starts: list[int], tol_px: float) -> float:
    """Every row starts at the same moment. A row whose trace begins off its baseline
    (baseline wander, a DC offset) is found late; the median of six rows with three late
    put a 6x2 page 176 ms off. The earliest start that another row confirms is taken."""
    s = sorted(starts)
    need = 2 if len(s) >= 2 else 1
    for i, x in enumerate(s):
        if sum(1 for y in s[i:] if y - x <= tol_px) >= need:
            return float(np.median([y for y in s[i:] if y - x <= tol_px]))
    return float(np.median(s))


def frame_page(layout: LayoutSpec, baselines: list[int], strokes: np.ndarray, px_per_mm: float,
               pulse_right: list[int]) -> PageFrame | None:
    rows = pick_rows(baselines, layout.n_trace_rows, layout.row_pitch_mm * px_per_mm)
    if rows is None:
        return None
    x0, x_end = trace_start(strokes, rows, px_per_mm, pulse_right)
    implied = None
    pitch = float(np.median(np.diff(rows))) if len(rows) > 1 else layout.row_pitch_mm * px_per_mm
    if layout.panel_gap_mm > 0:
        groups = strip_groups(strokes, rows, pitch, x0, x_end, px_per_mm, layout.panel_gap_mm)
        complete = len(groups) == layout.n_columns
        if groups:
            implied = float(np.median([b - a for a, b in groups])) / px_per_mm / layout.panel_s
        return PageFrame([float(r) for r in rows], x0, x_end, implied, pitch,
                         column_x0=[float(a) for a, _ in groups] if complete else None, groups_found=len(groups))
    if x_end is not None and x_end > x0:
        implied = (x_end - x0) / px_per_mm / layout.record_s
    return PageFrame([float(r) for r in rows], x0, x_end, implied, pitch)


def strip_groups(strokes: np.ndarray, rows: list[int], pitch_px: float, x0: float, x_end: float | None,
                 px_per_mm: float, gap_mm: float) -> list[tuple[float, float]]:
    """(start, end) of each run of trace ink across the rows, split where the paper is blank
    for at least half the nominal gap (the labels printed in a gap are not long strokes)."""
    if x_end is None or x_end <= x0:
        return []
    h, w = strokes.shape
    half = int(0.45 * pitch_px)
    cols = np.zeros(w, dtype=bool)
    for yb in rows:
        cols |= strokes[max(0, int(yb) - half):min(h, int(yb) + half + 1)].any(axis=0)
    a, b = int(round(x0)), int(round(x_end)) + 1
    ink = np.flatnonzero(cols[a:b]) + a
    if ink.size == 0:
        return []
    min_gap = 0.5 * gap_mm * px_per_mm
    breaks = np.flatnonzero(np.diff(ink) > min_gap)
    starts = [ink[0], *ink[breaks + 1]]
    ends = [*ink[breaks], ink[-1]]
    return [(float(s0), float(e0)) for s0, e0 in zip(starts, ends, strict=True)]


def panels(layout: LayoutSpec, frame: PageFrame, px_per_mm: float, speed_mm_s: float) -> list[Panel]:
    px_s = px_per_mm * speed_mm_s
    out = []
    for r, row in enumerate(layout.rows):
        for c, lead in enumerate(row):
            t0, t1 = c * layout.panel_s, (c + 1) * layout.panel_s
            xs = (frame.column_x0[c] if frame.column_x0 is not None
                  else frame.x0 + layout.column_offset_mm(c, speed_mm_s) * px_per_mm)
            out.append(Panel(lead, "panel", r, t0, t1, int(round(xs)), int(round(xs + layout.panel_s * px_s)),
                             frame.baselines[r]))
    for k, lead in enumerate(layout.rhythm):
        r = len(layout.rows) + k
        out.append(Panel(lead, "rhythm", r, 0.0, layout.record_s, int(round(frame.x0)),
                         int(round(frame.x0 + layout.record_s * px_s)), frame.baselines[r]))
    return out
