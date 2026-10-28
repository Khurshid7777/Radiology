"""Layout from the trace rows of a rectified page, checked against the registry (TZ §5.6).

Two independent cues:

* the number of trace rows equals a registry layout's rows (+ rhythm strips);
* the median baseline pitch (mm) matches a layout's ``row_pitch_mm``.

Both -> high confidence; the count alone -> medium (printers vary the pitch, e.g. 38 mm);
the pitch alone -> low confidence and a ``LAYOUT_UNCERTAIN`` warning (a stray
row from a shadow or a fold changes the count, rarely the median pitch); neither ->
unknown. Lead labels (OCR) join in M3.

Two layouts with the same number of rows (6x2 and 3x4+3rhythm, S2) are told apart by a
third cue: rhythm strips repeat their panel cells — the same lead over the same seconds,
so the same curve at the same x — and a 6x2 page repeats nothing. Undecided: the more
likely layout with ``LAYOUT_UNCERTAIN`` (the doctor confirms; TZ 18.5: ask, not guess).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.layout.registry import LayoutSpec, available_layouts
from ecgscan.layout.rows import long_stroke_mask, row_baselines

REPEAT_SAME = 0.8       # a rhythm strip's quarter this close to its panel cell is that cell repeated ...
REPEAT_MARGIN = 0.08    # ... and closer by this than the strip's other quarters are to that row
REPEAT_NONE = 0.7       # below this, or a margin below REPEAT_NO_MARGIN: no rhythm strips (6x2 photos <= -0.56)
REPEAT_NO_MARGIN = 0.03
REPEAT_SHIFT_MM = 1.5   # rows straightened from a photo can be this far apart in x
REPEAT_SMOOTH_MM = 0.3


@dataclass
class LayoutGuess:
    name: str | None
    confidence: float
    rows: int
    pitch_mm: float | None
    baselines: list[int]
    uncertain: bool
    repeats_open: bool = False      # 6x2 / 3x4+3rhythm left open by the repeats: only the doctor settles it


def merge_baselines(projection: list[int], pulse_bottoms: list[int], px_per_mm: float,
                    tol_mm: float = 4.0) -> list[int]:
    """Projection peaks plus calibration-pulse bottoms (a pulse stands on its row's baseline).
    Busy rows (a child's 180/min) have few flat segments and a weak projection peak; their
    pulse still marks them."""
    out = sorted(projection)
    for y in sorted(pulse_bottoms):
        if all(abs(y - b) > tol_mm * px_per_mm for b in out):
            out.append(int(y))
    return sorted(out)



def _row_curves(strokes: np.ndarray, baselines: list[int], half: int) -> np.ndarray:
    """Per row, the mean ink height above its baseline in every column (NaN: no ink)."""
    h, w = strokes.shape
    out = np.full((len(baselines), w), np.nan)
    for i, b in enumerate(baselines):
        y0, y1 = max(0, b - half), min(h, b + half)
        band = strokes[y0:y1]
        n = band.sum(axis=0)
        ys = np.arange(y0, y1, dtype=np.float64)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(n > 0, b - (band * ys).sum(axis=0) / n, np.nan)
    return out


def _filled(c: np.ndarray) -> np.ndarray:
    ok = np.isfinite(c)
    if ok.sum() < 2 or ok.all():
        return c
    out = c.copy()
    out[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok])
    return out


def _alike_near(a: np.ndarray, b: np.ndarray, shift: int, sigma: float) -> float:
    """``_alike`` at the best x shift up to ``shift`` px, both smoothed by ``sigma`` px."""
    from scipy.ndimage import gaussian_filter1d

    if len(a) <= 2 * shift + 20:
        return _alike(a, b)
    a, b = _filled(a), _filled(b)
    if sigma > 0:
        a, b = gaussian_filter1d(a, sigma), gaussian_filter1d(b, sigma)
    core = a[shift:len(a) - shift]
    return max(_alike(core, b[shift + d:len(b) - shift + d]) for d in range(-shift, shift + 1))


def _alike(a: np.ndarray, b: np.ndarray) -> float:
    """1 - rms(a - b) / rms(a) over the columns both have, each less its median: 1 for the
    same curve, ~0 or below for another lead (shape and amplitude must both agree)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return 0.0
    a, b = a[ok] - np.median(a[ok]), b[ok] - np.median(b[ok])
    ra = float(np.sqrt(np.mean(a ** 2)))
    return 0.0 if ra == 0 else 1.0 - float(np.sqrt(np.mean((a - b) ** 2))) / ra


def rhythm_repeats(strokes: np.ndarray, baselines: list[int], px_per_mm: float,
                   spec: LayoutSpec) -> tuple[float, float] | None:
    """How well the layout's rhythm strips repeat their panel cells. A strip of lead L shows,
    over the seconds of L's panel cell, the same curve as that cell: (worst such match, worst
    margin over the strip's other quarters against the same panel row). None: no strip has
    its lead in the panels, or the rows do not fit the layout."""
    n_panel, n_cols = len(spec.rows), spec.n_columns
    if not spec.rhythm or len(baselines) != n_panel + len(spec.rhythm):
        return None
    cells = [(n_panel + k, p, row.index(lead)) for k, lead in enumerate(spec.rhythm)
             for p, row in enumerate(spec.rows) if lead in row]
    if not cells:
        return None
    pitch = float(np.median(np.diff(baselines)))
    raw = _row_curves(strokes, baselines, max(2, int(0.5 * pitch)))
    inked = np.isfinite(raw)
    if not inked.any(axis=1).all():
        return None
    # Each row on its own time axis, start to end: a page straightened from a photo can be
    # sheared, its rows starting a few mm apart (27 px over six rows on a synthetic photo).
    spans = [(np.flatnonzero(r)[0], np.flatnonzero(r)[-1]) for r in inked]
    width = int(np.median([b - a for a, b in spans]))
    if width < 10 * px_per_mm * n_cols:
        return None
    curves = np.full((len(baselines), width), np.nan)
    for i, (a, b) in enumerate(spans):
        src = np.linspace(a, b, width)
        row = _filled(raw[i, a:b + 1])
        curves[i] = np.interp(src, np.arange(a, b + 1), row)
        curves[i][~np.interp(src, np.arange(a, b + 1), np.isfinite(raw[i, a:b + 1]).astype(float)).astype(bool)] = np.nan
    x0, seg = 0.0, width / n_cols

    shift, sigma = int(round(REPEAT_SHIFT_MM * px_per_mm)), REPEAT_SMOOTH_MM * px_per_mm

    def alike(r: int, p: int, c: int) -> float:
        a0, a1 = int(x0 + c * seg), int(x0 + (c + 1) * seg)
        return _alike_near(curves[r, a0:a1], curves[p, a0:a1], shift, sigma)

    same = [alike(r, p, c) for r, p, c in cells]
    margin = [alike(r, p, c) - max(alike(r, p, o) for o in range(n_cols) if o != c) for r, p, c in cells]
    return min(same), min(margin)


def _by_repeats(names: list[str], ink: np.ndarray, base: list[int], px_per_mm: float, stroke_min_height_mm: float,
                stroke_bridge_mm: float, exclude_boxes) -> tuple[str | None, bool]:
    """Among layouts with the same row count: (choice, settled)."""
    specs = available_layouts()
    with_rhythm = [x for x in names if specs[x].rhythm]
    without = [x for x in names if not specs[x].rhythm]
    if len(with_rhythm) != 1 or len(without) > 1:
        return None, False
    strokes = long_stroke_mask(ink, px_per_mm, min_height_mm=stroke_min_height_mm, bridge_mm=stroke_bridge_mm,
                               exclude_boxes=exclude_boxes)
    rep = rhythm_repeats(strokes, base, px_per_mm, specs[with_rhythm[0]])
    if rep is None:
        return (without[0] if without else None), False
    same, margin = rep
    if same >= REPEAT_SAME and margin >= REPEAT_MARGIN:
        return with_rhythm[0], True
    if (same < REPEAT_NONE or margin < REPEAT_NO_MARGIN) and without:
        return without[0], True
    return (with_rhythm[0] if not without or same >= 0.5 * (REPEAT_SAME + REPEAT_NONE) else without[0]), False


def detect_layout(ink: np.ndarray, px_per_mm: float, *, pitch_tol: float = 0.1,
                  pulse_bottoms: list[int] | None = None, stroke_min_height_mm: float = 3.0,
                  stroke_bridge_mm: float = 0.0, exclude_boxes: list | None = None) -> LayoutGuess:
    # A pulse bottom near a found row is that row's pulse, even when the printer floats it a
    # few mm off the baseline (a real strip: 4.1 mm below row II, read as a 4th row and a
    # 3x4+rhythm page, SD-059): no two rows are closer than the smallest registry pitch.
    near_mm = 0.35 * min(spec.row_pitch_mm for spec in available_layouts().values())
    base = merge_baselines(row_baselines(ink, px_per_mm, min_height_mm=stroke_min_height_mm, bridge_mm=stroke_bridge_mm,
                                         exclude_boxes=exclude_boxes),
                           pulse_bottoms or [], px_per_mm, tol_mm=max(4.0, near_mm))
    n = len(base)
    pitch = float(np.median(np.diff(base))) / px_per_mm if n > 1 else None
    by_count, by_pitch = [], []
    for spec in available_layouts().values():
        if spec.n_trace_rows == 1 and pulse_bottoms and len({int(y // (8 * px_per_mm)) for y in pulse_bottoms}) > 1:
            continue    # several pulses at different heights: not a single strip
        if spec.n_trace_rows == n:
            by_count.append(spec.name)
        if pitch is not None and abs(pitch - spec.row_pitch_mm) <= pitch_tol * spec.row_pitch_mm:
            by_pitch.append(spec.name)
    if len(by_count) > 1:
        # Same row count (6x2 / 3x4+3rhythm): the rhythm strips' repeats decide (S2).
        name, settled = _by_repeats(by_count, ink, base, px_per_mm, stroke_min_height_mm, stroke_bridge_mm, exclude_boxes)
        if name is not None:
            return LayoutGuess(name, 0.9 if settled and name in by_pitch else 0.8 if settled else 0.4, n, pitch, base,
                               not settled, repeats_open=not settled)
    both = [x for x in by_count if x in by_pitch]
    if len(both) == 1:
        return LayoutGuess(both[0], 0.9, n, pitch, base, False)
    if len(by_count) == 1:
        return LayoutGuess(by_count[0], 0.7, n, pitch, base, False)
    if len(by_pitch) == 1:
        return LayoutGuess(by_pitch[0], 0.4, n, pitch, base, True)
    return LayoutGuess(None, 0.0, n, pitch, base, True)
