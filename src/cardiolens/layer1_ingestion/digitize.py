"""Paper-ECG digitisation (TZ §4, v2 in TZ §17 "грязные фото").

Two levels:

* **Single lead** (MVP): ``render_ecg_image`` / ``digitize_image`` — a clean gridded
  strip, column-wise trace extraction.
* **Full 12-lead page** (v2): ``render_ecg_page`` draws the standard 3x4 layout
  (four 2.5-s columns: I/II/III, aVR/aVL/aVF, V1-V3, V4-V6) plus a 10-s lead-II
  rhythm strip; ``degrade_image`` produces photo-like defects for evaluation
  (rotation, blur, JPEG, shadow gradient, salt noise); ``digitize_page`` recovers
  every lead: de-skew from the grid lines (Hough), adaptive thresholding (robust
  to shading), row/column segmentation from trace projections with a
  layout-geometry fallback, per-panel trace **tracking** (skeleton + Viterbi over
  the ink runs, D47: a steep QRS edge stays in its own column and a neighbouring
  row's trace crossing the panel is rejected), then ``page_to_signal`` builds a
  10-s ``SignalRepresentation`` whose ``provenance.lead_windows`` says where each
  lead really has data.

Calibration is read from the page when not given (px/mm from the grid pitch and
the row pitch, mm/mV from the 1 mV pulse, D43); photo perspective is rectified
from the grid's vanishing points. ``provenance.confidence`` is the mean column
coverage over the panels, so a bad photo yields a low-confidence result rather
than an invented one (TZ §0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Provenance,
    SignalRepresentation,
)

# Standard 3x4 layout: row -> leads, column c covers [2.5c, 2.5(c+1)) s.
LAYOUT_3X4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
RHYTHM_LEAD = "II"
PANEL_SECONDS = 2.5
PAGE_MARGIN_MM = 8.0
ROW_HEIGHT_MM = 32.0
LABEL_OFFSET_MM = 2.0
_GRID_LIGHT = (230, 220, 220)
_GRID_DARK = (200, 180, 180)
_TRACE = (20, 20, 20)


# ----------------------------------------------------------------------------
# Single lead (MVP)
# ----------------------------------------------------------------------------
def _draw_grid(img: np.ndarray, px_per_mm: int) -> None:
    import cv2

    h, w = img.shape[:2]
    for x in range(0, w, px_per_mm):
        cv2.line(img, (x, 0), (x, h), _GRID_LIGHT, 1)
    for y in range(0, h, px_per_mm):
        cv2.line(img, (0, y), (w, y), _GRID_LIGHT, 1)
    for x in range(0, w, px_per_mm * 5):
        cv2.line(img, (x, 0), (x, h), _GRID_DARK, 1)
    for y in range(0, h, px_per_mm * 5):
        cv2.line(img, (0, y), (w, y), _GRID_DARK, 1)


def _draw_trace(img: np.ndarray, sig_uv: np.ndarray, fs: float, x0: int, y_center: int,
                px_per_s: float, px_per_uv: float) -> None:
    import cv2

    n = len(sig_uv)
    xs = (x0 + np.arange(n) / fs * px_per_s).astype(int)
    ys = (y_center - np.asarray(sig_uv, dtype=np.float64) * px_per_uv).astype(int)
    ys = np.clip(ys, 0, img.shape[0] - 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=_TRACE, thickness=2)


def render_ecg_image(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    path: str | Path,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    px_per_mm: int = 8,
    lead_name: str = "II",
) -> str:
    """Render one lead to a clean gridded PNG (a synthetic 'paper' strip)."""
    import cv2

    sig = np.asarray(signal_uv, dtype=np.float64)
    px_per_s = px_per_mm * mm_per_s
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    width = max(1, int(round(len(sig) / sampling_rate_hz * px_per_s)))
    height = int(round(30 * mm_per_mv / 10.0 * px_per_mm))
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
    _draw_trace(img, sig, sampling_rate_hz, 0, height // 2, px_per_s, px_per_uv)
    cv2.imwrite(str(path), img)
    return str(path)


def _trace_rows(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """Per-column median row of trace pixels inside a box (NaN where none)."""
    sub = mask[y0:y1, x0:x1]
    out = np.full(x1 - x0, np.nan)
    for i in range(x1 - x0):
        rows = np.flatnonzero(sub[:, i])
        if rows.size:
            out[i] = y0 + float(np.median(rows))
    return out


def _column_runs(col: np.ndarray, merge_gap: int = 3) -> list[tuple[int, int]]:
    """Vertical ink runs (top, bottom) of one column; runs separated by a gap of
    ``merge_gap`` px or less are one run (a thresholded stroke breaks at speckle)."""
    d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
    runs = list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def thin_mask(mask: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """Zhang-Suen thinning of an ink mask to a 1-px skeleton (vectorised).

    A rendered or printed trace is 3-4 px thick after thresholding, so a steep
    QRS edge occupies 3-4 columns instead of one and the per-column median widens
    the QRS by a column or two per edge (+24 ms at 150 px/s, D47). The skeleton
    puts the edge back into its own column. Its line ends are eroded by a few
    pixels and its apex sits off-centre on a thick spike — both handled by
    ``_trace_rows_tracked`` (raw-ink fallback at ends, apex from the raw ink).
    """
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                c1, c2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            rm = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


TRACK_BASELINE_WEIGHT = 0.25   # pull towards the panel's own baseline, per column, in row heights
TRACK_EXTEND = 0.5             # panel box extended by this fraction of its height up and down
TRACK_TRIM_COLUMNS = 3         # panel-edge columns ignored (the neighbour panel's trace touches them)
TRACK_MAX_JUMP = 0.25          # a run farther than this (in box heights) from the trace on both sides is not the trace
TRACK_EDGE_SCAN = 12           # columns from each panel end checked against the median row (80 ms at 150 px/s)
TRACK_TALL = 3.0               # a run taller than this many stroke widths is a steep stroke, not flat trace
TRACK_TURN_PX = 3              # the raw ink must reach this far beyond both sides to mark a turning column
TRACK_FLANK = 2                # columns on each side of an apex read from the raw run midpoint


def _trace_rows_tracked(mask: np.ndarray, thin: np.ndarray, x0: int, x1: int, y0: int, y1: int,
                        baseline_weight: float = TRACK_BASELINE_WEIGHT) -> np.ndarray:
    """Per-column trace row by tracking, not by the median of all ink (D47).

    * The box is extended by ``TRACK_EXTEND`` of its height above and below, so
      a deep S wave that leaves its row is followed instead of clipped.
    * Candidates per column are the runs of the skeleton (``thin_mask``): the
      skeleton keeps a steep edge in its own column, where the raw stroke smears
      it over 3-4 columns. Where the skeleton is empty and no skeleton ink sits
      in a neighbour column (line ends, eroded by thinning) the raw runs serve;
      next to a skeleton column an empty column is the lateral spread of a
      steep stroke and stays empty.
    * A Viterbi pass picks one run per column minimising continuity cost
      (distance from the previous column's value to the run) plus a weak pull
      towards the panel's own baseline, so a neighbouring lead's trace crossing
      the box loses even where it is the closest ink; forced choices (a run
      far from the trace on both sides, or at a panel end) are dropped.
    * The skeleton's apex is off-centre on a thick spike and stops short of the
      tip; the raw ink is symmetric, so a turning column is where the raw run
      top (bottom) is a local extreme along the path — the middle of a tied
      plateau — and reads the raw extreme minus the stroke radius.
    Columns without usable ink stay NaN (interpolated by the caller).
    """
    h = y1 - y0
    ext = int(TRACK_EXTEND * h)
    ya, yb = max(0, y0 - ext), min(mask.shape[0], y1 + ext)
    sub, sub_thin = mask[ya:yb, x0:x1], thin[ya:yb, x0:x1]
    centre = (y0 + y1) / 2.0 - ya
    n = x1 - x0
    raw_cols = [_column_runs(sub[:, i]) for i in range(n)]
    thin_cols = [_column_runs(sub_thin[:, i]) for i in range(n)]
    heights = [b - a + 1 for runs in raw_cols for a, b in runs]
    stroke = float(np.clip(np.median(heights), 1.0, max(2.0, h / 24.0))) if heights else 2.0
    cols: list[list[tuple[int, int]]] = []
    for i in range(n):
        if thin_cols[i]:
            cols.append(thin_cols[i])
        elif thin_cols[max(0, i - 1)] or thin_cols[min(n - 1, i + 1)]:
            cols.append([])
        else:
            cols.append(raw_cols[i])
    # --- Viterbi over the runs -------------------------------------------------
    prev_cost: list[float] | None = None
    prev_val: list[float] = []
    last = -1
    back: list[tuple[list[int], list[float]] | None] = []
    for i in range(n):
        runs = cols[i]
        if not runs:
            back.append(None)
            continue
        mids = [(a + b) / 2.0 for a, b in runs]
        cost = [baseline_weight * abs(m - centre) / h for m in mids]
        bp = [-1] * len(runs)
        if prev_cost is not None:
            gap = float(i - last)
            for j, (a, b) in enumerate(runs):
                best, bj = np.inf, -1
                for k, pv in enumerate(prev_val):
                    d = 0.0 if a <= pv <= b else min(abs(pv - a), abs(pv - b))
                    c = prev_cost[k] + d / h / gap
                    if c < best:
                        best, bj = c, k
                cost[j] += best
                bp[j] = bj
        prev_cost, prev_val, last = cost, mids, i
        back.append((bp, mids))
    out = np.full(n, np.nan)
    if prev_cost is None:
        return out
    chosen: dict[int, tuple[int, int]] = {}
    j, i = int(np.argmin(prev_cost)), last
    while i >= 0 and j >= 0:
        bp, mids = back[i]  # type: ignore[misc]
        chosen[i] = cols[i][j]
        out[i] = ya + mids[j]
        j = bp[j]
        i -= 1
        while i >= 0 and back[i] is None:
            i -= 1
    found = np.isfinite(out)   # columns that held usable ink: the coverage the caller reports
    # --- Forced choices: a run the trace cannot have reached ------------------
    # Where the own trace offers no candidate but a foreign run exists, the path
    # had to take it. A run farther than TRACK_MAX_JUMP of the box height from
    # the values on both sides is such a choice; a genuine R spike is not, its
    # tall run contains the neighbouring values.
    for _ in range(3):
        keys = sorted(chosen)
        dropped = False
        for idx, k in enumerate(keys):
            a, b = chosen[k]
            nbs = [keys[q] for q in (idx - 1, idx + 1) if 0 <= q < len(keys) and abs(keys[q] - k) <= 3]
            if not nbs:
                continue
            far = 0
            for nb in nbs:
                v = out[nb] - ya
                d = 0.0 if a <= v <= b else min(abs(v - a), abs(v - b))
                far += d > TRACK_MAX_JUMP * h
            if far == len(nbs):
                out[k] = np.nan
                del chosen[k]
                dropped = True
        if not dropped:
            break
    # --- Panel ends: the own trace stops a few columns short of the edge --------
    finite = np.flatnonzero(np.isfinite(out))
    if finite.size:
        med = float(np.median(out[finite]))
        for idx in (finite, finite[::-1]):
            for i in idx[:TRACK_EDGE_SCAN]:
                if abs(out[i] - med) <= TRACK_MAX_JUMP * h:
                    break
                out[i] = np.nan
                chosen.pop(int(i), None)
    # --- Turning columns from the raw ink profile ------------------------------
    tall = TRACK_TALL * stroke
    keys = sorted(chosen)
    raw_ext: dict[int, tuple[int, int]] = {}
    for k in keys:
        a, b = chosen[k]
        mid = (a + b) / 2.0
        hit = [r for r in raw_cols[k] if r[0] - 1 <= mid <= r[1] + 1]
        raw_ext[k] = hit[0] if hit else (a, b)
    tops = {k: raw_ext[k][0] for k in keys}
    bots = {k: raw_ext[k][1] for k in keys}
    flanks: set[int] = set()
    apexes: set[int] = set()
    for idx, k in enumerate(keys):
        ra, rb = raw_ext[k]
        if rb - ra + 1 < tall:
            continue
        left = [keys[q] for q in range(max(0, idx - 3), idx)]
        right = [keys[q] for q in range(idx + 1, min(len(keys), idx + 4))]
        if not left or not right:
            continue
        # upward apex: the top is at least as high as the immediate neighbours and
        # clearly higher than the outer ones; ties are one plateau — take its middle
        if (ra <= tops[left[-1]] and ra <= tops[right[0]]
                and ra < min(tops[left[0]], tops[right[-1]]) - TRACK_TURN_PX):
            plateau = [q for q in left[-1:] + [k] + right[:1] if abs(tops[q] - ra) <= 1]
            if plateau[len(plateau) // 2] == k:
                out[k] = ya + ra + stroke / 2.0
                apexes.add(k)
                flanks.update(q for q in left[-TRACK_FLANK:] + right[:TRACK_FLANK] if q != k)
        elif (rb >= bots[left[-1]] and rb >= bots[right[0]]
                and rb > max(bots[left[0]], bots[right[-1]]) + TRACK_TURN_PX):
            plateau = [q for q in left[-1:] + [k] + right[:1] if abs(bots[q] - rb) <= 1]
            if plateau[len(plateau) // 2] == k:
                out[k] = ya + rb - stroke / 2.0
                apexes.add(k)
                flanks.update(q for q in left[-TRACK_FLANK:] + right[:TRACK_FLANK] if q != k)
    # On the flanks of a thick spike the skeleton runs sit near the blob's base
    # (the medial axis of a filled spike is one line up its middle), reading the
    # flank far too low under blur. There the raw run's midpoint is the fair
    # estimate: its lateral-spread bias matters at the QRS onset/offset columns,
    # which lie farther out, not next to the apex. A tip two columns wide is two
    # adjacent apexes (each is the middle of its own three-column plateau); they
    # must not demote each other to flanks — that read a 26-px R as 17 px (D52).
    for q in flanks - apexes:
        ra, rb = raw_ext[q]
        if rb - ra + 1 >= tall:
            out[q] = ya + (ra + rb) / 2.0
    t = TRACK_TRIM_COLUMNS
    if t and n > 2 * t:
        out[:t] = np.nan
        out[-t:] = np.nan
    # Columns dropped by design (trimmed, rejected) are not missing ink: interpolate
    # them here so the caller's coverage counts only the columns that had no trace.
    ok = np.isfinite(out)
    fill = found & ~ok
    if fill.any() and ok.sum() >= 2:
        idx = np.arange(n)
        out[fill] = np.interp(idx[fill], idx[ok], out[ok])
    return out


def _fill_gaps(trace: np.ndarray) -> tuple[np.ndarray, float]:
    found = np.isfinite(trace)
    if found.sum() < 2:
        raise ValueError("no trace detected in image")
    idx = np.arange(len(trace))
    filled = trace.copy()
    filled[~found] = np.interp(idx[~found], idx[found], trace[found])
    return filled, float(found.mean())


def digitize_image(
    path: str | Path,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    px_per_mm: int = 8,
    lead_name: str = "II",
    dark_threshold: int = 110,
) -> SignalRepresentation:
    """Recover a single-lead ``SignalRepresentation`` from a gridded strip (TZ §4)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    height, width = gray.shape
    px_per_s = px_per_mm * mm_per_s
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    trace, coverage = _fill_gaps(_trace_rows(gray < dark_threshold, 0, width, 0, height))
    signal_uv = (height // 2 - trace) * uv_per_px
    fs = float(px_per_s)
    return SignalRepresentation(
        signal=signal_uv.astype(np.float32).reshape(1, -1), sampling_rate_hz=fs,
        lead_names=[lead_name], duration_s=float(width / fs), source="image",
        provenance=Provenance(digitized=True, confidence=round(coverage, 3)),
    )


# ----------------------------------------------------------------------------
# 12-lead page (v2)
# ----------------------------------------------------------------------------
@dataclass
class PageGeometry:
    px_per_mm: int
    mm_per_s: float
    mm_per_mv: float
    width: int
    height: int
    margin: int
    row_h: int
    panel_w: int
    x0: int = 0          # left edge of the trace area (margin + pulse area)

    @property
    def px_per_s(self) -> float:
        return self.px_per_mm * self.mm_per_s

    @property
    def px_per_uv(self) -> float:
        return self.px_per_mm * self.mm_per_mv / 1000.0

    def panel_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        """(x0, x1, y0, y1) of a layout panel; row 3 = rhythm strip (full width)."""
        y0 = self.margin + row * self.row_h
        left = self.x0 or self.margin
        if row == 3:
            return left, left + 4 * self.panel_w, y0, y0 + self.row_h
        x0 = left + col * self.panel_w
        return x0, x0 + self.panel_w, y0, y0 + self.row_h


PULSE_AREA_MM = 8.0     # left of the traces when a calibration pulse is drawn
PULSE_WIDTH_MM = 5.0    # 200 ms at 25 mm/s
PULSE_HEIGHT_MV = 1.0


def page_geometry(px_per_mm: int = 6, mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                  pulse: bool = False) -> PageGeometry:
    margin = int(PAGE_MARGIN_MM * px_per_mm)
    row_h = int(ROW_HEIGHT_MM * px_per_mm)
    panel_w = int(PANEL_SECONDS * mm_per_s * px_per_mm)
    x0 = margin + (int(PULSE_AREA_MM * px_per_mm) if pulse else 0)
    return PageGeometry(px_per_mm, mm_per_s, mm_per_mv, x0 + margin + 4 * panel_w,
                        2 * margin + 4 * row_h, margin, row_h, panel_w, x0)


def render_ecg_page(
    sig: SignalRepresentation, path: str | Path, *, px_per_mm: int = 6,
    mm_per_s: float = 25.0, mm_per_mv: float = 10.0, calibration_pulse: bool = False,
) -> str:
    """Render a 12-lead 3x4 page with a lead-II rhythm strip (10 s of data needed).

    With ``calibration_pulse`` a 1 mV x 200 ms rectangle precedes each row, as on
    printed ECGs; ``digitize_page`` reads mm/mV from it.
    """
    import cv2

    g = page_geometry(px_per_mm, mm_per_s, mm_per_mv, pulse=calibration_pulse)
    img = np.full((g.height, g.width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
    if calibration_pulse:
        pw, ph = int(PULSE_WIDTH_MM * px_per_mm), int(PULSE_HEIGHT_MV * mm_per_mv * px_per_mm)
        for row in range(4):
            _x0, _x1, y0, y1 = g.panel_box(row, 0)
            base = (y0 + y1) // 2
            xa = g.margin + int(1.0 * px_per_mm)
            pts = np.array([[xa, base], [xa, base - ph], [xa + pw, base - ph], [xa + pw, base]], dtype=np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False, color=_TRACE, thickness=2)
    fs = sig.sampling_rate_hz
    seg = int(PANEL_SECONDS * fs)
    for row, leads in enumerate(LAYOUT_3X4):
        for col, lead in enumerate(leads):
            x0, _x1, y0, y1 = g.panel_box(row, col)
            data = sig.lead(lead)[col * seg:(col + 1) * seg]
            _draw_trace(img, data, fs, x0, (y0 + y1) // 2, g.px_per_s, g.px_per_uv)
            cv2.putText(img, lead, (x0 + int(LABEL_OFFSET_MM * px_per_mm), y0 + int(4 * px_per_mm)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.08 * px_per_mm, _TRACE, 1, cv2.LINE_AA)
    x0, _x1, y0, y1 = g.panel_box(3, 0)
    _draw_trace(img, sig.lead(RHYTHM_LEAD)[: int(4 * seg)], fs, x0, (y0 + y1) // 2, g.px_per_s, g.px_per_uv)
    cv2.putText(img, RHYTHM_LEAD, (x0 + int(LABEL_OFFSET_MM * px_per_mm), y0 + int(4 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.08 * px_per_mm, _TRACE, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


def degrade_image(
    src: str | Path, dst: str | Path, *, rotation_deg: float = 0.0, blur_px: int = 0,
    jpeg_quality: int | None = None, shadow: float = 0.0, salt_noise: float = 0.0, seed: int = 0,
    perspective: float = 0.0, vignette: float = 0.0, texture: float = 0.0,
) -> str:
    """Write a photo-like degraded copy.

    rotation, blur, JPEG, shadow gradient, salt noise, plus the phone-photo
    defects: ``perspective`` (corners displaced by up to this fraction of the
    size), ``vignette`` (radial darkening towards the corners), ``texture``
    (low-frequency paper/lighting unevenness).
    """
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if perspective > 0:
        d = perspective * np.array([w, h], dtype=np.float32)
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = src_pts + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * d
        dst_pts -= dst_pts.min(axis=0)
        hm = cv2.getPerspectiveTransform(src_pts, dst_pts)
        out_w, out_h = int(dst_pts[:, 0].max()) + 1, int(dst_pts[:, 1].max()) + 1
        img = cv2.warpPerspective(img, hm, (out_w, out_h), borderValue=(255, 255, 255))
        h, w = img.shape[:2]
    if rotation_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    if shadow > 0:
        ramp = np.linspace(1.0 - shadow, 1.0, w, dtype=np.float32)[None, :, None]
        img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    if vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        gain = (1.0 - vignette * rr ** 2).astype(np.float32)[:, :, None]
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if texture > 0:
        noise = rng.normal(0, 1, (h // 32 + 1, w // 32 + 1)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        gain = (1.0 + texture * noise / 3.0).astype(np.float32)[:, :, None]
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if blur_px and blur_px > 0:
        k = blur_px * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if salt_noise > 0:
        mask = rng.random((h, w)) < salt_noise
        img[mask] = rng.choice([0, 255], size=(int(mask.sum()), 1))
    dst = str(dst)
    if jpeg_quality is not None:
        if not dst.lower().endswith((".jpg", ".jpeg")):
            dst = str(Path(dst).with_suffix(".jpg"))
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        cv2.imwrite(dst, img)
    return dst


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle (degrees) from long near-horizontal grid/trace lines (Hough)."""
    import cv2

    edges = cv2.Canny(gray, 50, 150)
    min_len = max(50, gray.shape[1] // 3)
    # maxLineGap small: JPEG 8x8 block edges are axis-aligned and would otherwise
    # chain into long "lines" that outvote the real grid (skew read as 0.3 deg
    # instead of 3 deg on a q95 JPEG).
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=min_len, maxLineGap=5)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= 15:
            angles.append(ang)
    return float(np.median(angles)) if angles else 0.0


TRACE_RATIO = 0.6            # pixel darker than this fraction of the local mean = ink
MIN_COMPONENT_WIDTH = 0.05   # of the panel width: shorter blobs are labels / specks


def trace_mask(gray: np.ndarray, px_per_mm: int) -> np.ndarray:
    """Ink pixels, robust to shading and to the printed grid.

    Ratio to the local mean brightness (8 mm window) instead of a constant offset:
    the trace (~8 % of the paper brightness) passes under any shadow, the 5-mm grid
    lines (~75 %) never do.
    """
    import cv2

    k = 2 * int(4 * px_per_mm) + 1
    local = cv2.blur(gray.astype(np.float32), (k, k)) + 1.0
    # No morphological opening: it fragments the 2-px trace into blobs that the
    # component filter would then discard.
    return gray.astype(np.float32) < TRACE_RATIO * local


def _keep_long_components(sub: np.ndarray, min_width: int) -> np.ndarray:
    """Drop ink blobs narrower than ``min_width`` (lead labels, specks, grid stubs)."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] >= min_width:
            keep[i] = True
    return keep[labels]


def _bands(profile: np.ndarray, n: int, min_gap: int) -> list[tuple[int, int]]:
    """The ``n`` widest runs of a projection profile above 10 % of its max."""
    thr = 0.10 * float(profile.max()) if profile.max() > 0 else 0.0
    on = profile > thr
    runs, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    merged: list[tuple[int, int]] = []
    for a, b in runs:  # close small gaps (letters, thin breaks)
        if merged and a - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged.sort(key=lambda r: r[1] - r[0], reverse=True)
    return sorted(merged[:n])


GRID_PITCH_RANGE_PX = (3, 64)   # periods searched (px): 1-mm pitch up to ~12 px, its 5-mm line period up to 64


GRID_TILE_PX = 256           # pitch is estimated per tile (a residual tilt smears a whole-page projection)


def _ac_peaks(prof: np.ndarray) -> list[float]:
    """Prominent autocorrelation peaks (px, sub-pixel) of a de-trended 1-D profile."""
    import cv2

    prof = prof.astype(np.float32)
    if prof.size < 3 * GRID_PITCH_RANGE_PX[1]:
        return []
    trend = cv2.blur(prof.reshape(1, -1), (61, 1)).ravel()   # ksize = (width, height)
    x = prof - trend
    x -= x.mean()
    if not np.any(x):
        return []
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = GRID_PITCH_RANGE_PX
    out = []
    for k in range(max(lo, 4), min(hi, len(ac) - 4)):
        if (ac[k] > 0.10 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]
                and ac[k] - max(ac[k - 3], ac[k + 3]) > 0.05):
            y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
            denom = (y0 - 2 * y1 + y2)
            out.append(float(k + (0.5 * (y0 - y2) / denom if denom != 0 else 0.0)))
    return out


def _vote_pitch(peaks: list[float]) -> float | None:
    """Grid pitch from the pooled autocorrelation peaks of many tiles.

    The 5-mm line period P (most robust to blur) is chosen among the periods
    >= 15 px by harmonic support: the true P is backed by peaks at the 1-mm
    harmonics P/5, 2P/5, 3P/5, 4P/5, while the sum/difference aliases (P +/- p)
    are not. Pitch = P / 5. Without any period >= 15 px the smallest supported
    small period is returned.
    """
    if len(peaks) < 3:
        return None
    arr = np.asarray(peaks)

    def count(c: float) -> int:
        return int(np.sum(np.abs(arr - c) <= 0.75))

    def refined(c: float) -> float:
        sel = arr[np.abs(arr - c) <= 0.75]
        return float(np.median(sel)) if sel.size else c

    big = [(float(c), count(c)) for c in range(15, GRID_PITCH_RANGE_PX[1] + 1) if count(c) >= 2]
    if big:
        thr = max(2, 0.3 * max(n for _, n in big))
        cands = [(n + sum(count(k * c / 5.0) for k in (1, 2, 3, 4)), c) for c, n in big if n >= thr]
        top = max(sc for sc, _ in cands)
        best = min(c for sc, c in cands if sc >= 0.8 * top)
        # 2P carries the same harmonics as P; a period at P/10 (the true 1-mm
        # pitch) betrays it. With no 1-mm support at all (blurred photo) a
        # period-P grid still shows a peak at P, so the smallest strong period wins.
        if best / 2 >= 15 and count(best / 10) >= 2 and count(best / 5) >= 2:
            best = best / 2
        elif count(best / 5) < 2:
            best = min(c for _, c in cands)
        return refined(best) / 5.0
    centres = np.arange(4, 15)
    counts = np.array([count(c) for c in centres])
    if counts.max() < 2:
        return None
    thr = max(2, 0.3 * counts.max())
    small = [float(c) for c, n in zip(centres, counts) if n >= thr]
    return refined(min(small)) if small else None


def estimate_grid_pitch_xy(gray: np.ndarray) -> tuple[float | None, float | None]:
    """(pitch along x, pitch along y) in px per 1 mm from the grid's periodicity:
    autocorrelation peaks of de-trended darkness and edge profiles of every tile,
    pooled per axis and voted (5-mm period / 5 preferred, 1-mm period as backup)."""
    dark = 255.0 - gray.astype(np.float32)
    gx = np.abs(np.diff(dark, axis=1, prepend=dark[:, :1]))
    gy = np.abs(np.diff(dark, axis=0, prepend=dark[:1, :]))
    h, w = dark.shape
    t = GRID_TILE_PX
    px_peaks: list[float] = []
    py_peaks: list[float] = []
    for y in range(0, max(1, h - t + 1), t):
        for x in range(0, max(1, w - t + 1), t):
            tile = (slice(y, min(h, y + t)), slice(x, min(w, x + t)))
            px_peaks += _ac_peaks(dark[tile].mean(axis=0)) + _ac_peaks(gx[tile].mean(axis=0))
            py_peaks += _ac_peaks(dark[tile].mean(axis=1)) + _ac_peaks(gy[tile].mean(axis=1))
    return _vote_pitch(px_peaks), _vote_pitch(py_peaks)


def estimate_grid_pitch(gray: np.ndarray) -> float | None:
    """Pixels per 1 mm (median of the x and y estimates), None when neither axis
    shows a clear period."""
    px, py = estimate_grid_pitch_xy(gray)
    if px is not None and py is not None:
        # Row profiles are polluted by the (horizontal) traces; trust the column
        # estimate when the two disagree by more than 10 %.
        return float(np.median([px, py])) if abs(px - py) <= 0.1 * px else px
    return px if px is not None else py


def detect_calibration_pulse(mask: np.ndarray, x0: int, y0: int, y1: int, px_per_mm: float) -> tuple[float, int] | None:
    """(pulse height px, right edge x) of a 1-mV rectangle at the row's left edge.

    The rectangle's TOP edge is a horizontal ink run about PULSE_WIDTH_MM wide — a
    QRS never draws one — with vertical edges at both ends reaching down to the
    baseline; the height is measured from the top run to the lower end of those edges.
    """
    span = int((PULSE_AREA_MM + PULSE_WIDTH_MM + 4) * px_per_mm)
    sub = mask[y0:y1, x0: x0 + span]
    if sub.size == 0:
        return None
    min_w = int(0.7 * PULSE_WIDTH_MM * px_per_mm)
    best = None
    for y in range(sub.shape[0]):
        cols = np.flatnonzero(sub[y])
        if cols.size < min_w:
            continue
        # longest contiguous run in this row
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.r_[0, breaks + 1]; ends = np.r_[breaks, cols.size - 1]
        k = int(np.argmax(ends - starts))
        a, b = int(cols[starts[k]]), int(cols[ends[k]])
        if b - a + 1 >= min_w and (best is None or y < best[0]):
            best = (y, a, b)
    if best is None:
        return None
    y_top, a, b = best
    # Stroke thickness of the top edge (rows of ink below y_top at mid-run).
    mid = (a + b) // 2
    thick = 1
    while y_top + thick < sub.shape[0] and sub[y_top + thick, mid]:
        thick += 1
    edges = []
    for c in (a, a + 1, b - 1, b):
        rows = np.flatnonzero(sub[y_top:, c])
        if rows.size:
            edges.append(int(rows[-1]))
    if len(edges) < 2:
        return None
    # Height between the stroke centres of the top edge and the baseline.
    height = float(np.median(edges)) - thick + 1
    if height < 4 * px_per_mm:
        return None
    return height, x0 + b + 1


DEWARP_MIN_TILT_DEG = 0.3    # angle change across the page below this = no perspective (a rotation is not one)
DEWARP_MIN_CORR = 0.5        # ... and the angle must correlate with position (not just scatter)


def _vanishing_point(segments: np.ndarray, horizontal: bool) -> np.ndarray | None:
    """Vanishing point (homogeneous) of one family of grid lines, from how their
    angle changes with position across the page; None when they stay parallel."""
    if len(segments) < 8:
        return None
    x1, y1, x2, y2 = segments.T
    ang = np.arctan2(y2 - y1, x2 - x1)
    if not horizontal:
        ang = ang - np.sign(ang) * np.pi / 2  # vertical lines measured from the vertical
    pos = (y1 + y2) / 2 if horizontal else (x1 + x2) / 2
    a, b = np.polyfit(pos, ang, 1)
    p0, p1 = float(pos.min()), float(pos.max())
    if abs(np.degrees(a * (p1 - p0))) < DEWARP_MIN_TILT_DEG:
        return None
    # A perspective makes the angle change *systematically* with position; a
    # rotation or noisy segments give a slope without correlation.
    if pos.std() == 0 or ang.std() == 0 or abs(np.corrcoef(pos, ang)[0, 1]) < DEWARP_MIN_CORR:
        return None
    def line(p, theta):
        # homogeneous line through the family's mean point at position p with angle theta
        if horizontal:
            pt = np.array([float(np.mean((x1 + x2) / 2)), p, 1.0]); d = np.array([np.cos(theta), np.sin(theta), 0.0])
        else:
            pt = np.array([p, float(np.mean((y1 + y2) / 2)), 1.0]); d = np.array([-np.sin(theta), np.cos(theta), 0.0])
        return np.cross(pt, pt + d)
    return np.cross(line(p0, a * p0 + b), line(p1, a * p1 + b))


def _grid_segments(gray: np.ndarray) -> np.ndarray | None:
    import cv2

    edges = cv2.Canny(gray, 40, 120)
    w = gray.shape[1]
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=max(40, w // 5), maxLineGap=15)
    return None if segs is None else np.asarray(segs, dtype=np.float64).reshape(-1, 4)


def _line_angle_spread(gray: np.ndarray) -> float:
    """How far the grid's line families are from parallel (degrees): the spread of
    the near-horizontal and near-vertical segment angles. Smaller = flatter page."""
    segs = _grid_segments(gray)
    if segs is None or len(segs) < 8:
        return float("inf")
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    h = ang[np.abs(ang) < 10]
    v = np.abs(ang[np.abs(np.abs(ang) - 90) < 10]) - 90
    parts = [np.std(a) for a in (h, v) if len(a) >= 4]
    return float(np.mean(parts)) if parts else float("inf")


QUAD_MIN_AREA = 0.3            # the page quadrilateral must cover this fraction of the image
QUAD_MIN_PERSPECTIVE = 0.01    # projective scale change across the page below this: not a perspective (a rotation is affine)
QUAD_TEXTURE_MIN = 6.0         # grey-level residual after a local mean that counts as grid texture


def page_quadrilateral(gray: np.ndarray) -> np.ndarray | None:
    """The four corners (tl, tr, br, bl) of the printed grid area, or None.

    A photographed page is a quadrilateral against its background; the grid
    texture (light lines every mm) fills it and nothing else. The convex hull of
    the grid-textured region, reduced to four points, gives the corners directly —
    unlike the vanishing-point fit this needs no near-parallel line families and
    survives strong perspective, as long as the whole page is in the frame.
    """
    import cv2

    h, w = gray.shape
    # Fine structure (grid lines, trace) against a smooth background: the grid
    # is a periodic pattern a few pixels wide, so the residual after a local mean
    # marks it; a vignette, a shadow or the table behind the page are smooth and
    # do not. A plain intensity window would take a darkened corner for paper.
    g = gray.astype(np.float32)
    k_hp = 2 * int(min(h, w) / 120) + 1
    resid = np.abs(g - cv2.blur(g, (k_hp, k_hp)))
    ink = (resid > QUAD_TEXTURE_MIN).astype(np.uint8)
    k = max(9, int(min(h, w) / 60))
    filled = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < QUAD_MIN_AREA * h * w:
        return None
    pts = cv2.findNonZero((labels == big).astype(np.uint8))
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        return None
    # The polygon's corners are hull vertices moved by the morphology (up to k/2
    # px): refine each side by a least-squares line through the hull points near
    # it and take the corners as the intersections of adjacent sides.
    hp = hull.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        dist = np.abs((hp - a) @ nrm)
        near = hp[dist < max(3.0, 0.75 * k)]
        if len(near) < 2:
            near = np.array([a, b])
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((float(x0), float(y0), float(vx), float(vy)))
    refined = []
    for i in range(4):
        x1, y1, u1, v1 = lines[i - 1]
        x2, y2, u2, v2 = lines[i]
        det = u1 * v2 - v1 * u2
        if abs(det) < 1e-9:
            refined.append(quad[i])
            continue
        s_ = ((x2 - x1) * v2 - (y2 - y1) * u2) / det
        refined.append(np.array([x1 + s_ * u1, y1 + s_ * v1]))
    quad = np.array(refined)
    # Order: top-left, top-right, bottom-right, bottom-left.
    s = quad.sum(axis=1)
    d = quad[:, 0] - quad[:, 1]
    tl, br = quad[np.argmin(s)], quad[np.argmax(s)]
    tr, bl = quad[np.argmax(d)], quad[np.argmin(d)]
    out = np.array([tl, tr, br, bl])
    if len({tuple(p) for p in out}) < 4:
        return None
    # Page-like: every corner angle between 50 and 130 degrees.
    for i in range(4):
        a, b, c = out[i - 1], out[i], out[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        if not -0.64 < cosang < 0.64:
            return None
    return out


def _snap_quad_to_grid_lines(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Refine a coarse page quadrilateral (hull corners are off by up to ~7 px,
    which leaves a residual perspective the downstream steps cannot absorb) by
    snapping each side to the outermost grid line: a Hough line search in a band
    around the side, keeping the longest line nearly parallel to it, then
    intersecting adjacent sides. Sides without such a line keep their estimate."""
    import cv2

    h, w = gray.shape
    band = max(8, int(min(h, w) / 80))
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = float(np.linalg.norm(d)) + 1e-9
        u = d / length
        nrm = np.array([-u[1], u[0]])
        poly = np.array([a + nrm * band, b + nrm * band, b - nrm * band, a - nrm * band]).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)     # paper grain must not become edges
        edges = cv2.Canny(smooth, 30, 90) * mask
        found = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=int(0.15 * length),
                                minLineLength=int(0.4 * length), maxLineGap=int(0.05 * length))
        best = None
        if found is not None:
            side_ang = np.arctan2(u[1], u[0])
            for x1, y1, x2, y2 in np.asarray(found).reshape(-1, 4):
                ang = np.arctan2(y2 - y1, x2 - x1)
                dang = abs((ang - side_ang + np.pi / 2) % np.pi - np.pi / 2)
                if dang >= np.radians(2.0):
                    continue
                # Among long, parallel candidates take the one nearest the coarse
                # side, so all four sides snap to the same (outermost) grid line.
                # Tighter acceptance (both ends inside the band, 60 % of the side,
                # corner move within one band) was measured on 100 pages and
                # rejected: it lost the 2 % photo (r 0.65 -> 0.59) and 5 %
                # perspective for a gain only on rotated, vignetted 3.5 % photos,
                # where no snap variant beats the coarse quadrilateral (D57).
                off = abs(float((np.array([x1, y1]) - a) @ nrm))
                if best is None or off < best[0]:
                    best = (off, float(x1), float(y1), float(x2), float(y2))
        if best is None:
            lines.append((a[0], a[1], u[0], u[1]))
        else:
            _, x1, y1, x2, y2 = best
            v = np.array([x2 - x1, y2 - y1]) / (np.hypot(x2 - x1, y2 - y1) + 1e-9)
            lines.append((x1, y1, v[0], v[1]))
    out = []
    for i in range(4):
        x1, y1, u1, v1 = lines[i - 1]
        x2, y2, u2, v2 = lines[i]
        det = u1 * v2 - v1 * u2
        if abs(det) < 1e-9:
            out.append(quad[i])
            continue
        s_ = ((x2 - x1) * v2 - (y2 - y1) * u2) / det
        pt = np.array([x1 + s_ * u1, y1 + s_ * v1])
        # A snapped corner must stay near the coarse one (the band is narrow).
        out.append(pt if np.linalg.norm(pt - quad[i]) <= 3 * band else quad[i])
    return np.array(out)


def _dewarp_by_quadrilateral(gray: np.ndarray) -> np.ndarray | None:
    """Map the page quadrilateral onto an axis-aligned rectangle of the same
    mean side lengths (the aspect is corrected afterwards from the grid pitch)."""
    import cv2

    quad = page_quadrilateral(gray)
    if quad is None:
        return None
    h, w = gray.shape

    def homography(q):
        tl, tr, br, bl = q
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        hm = cv2.getPerspectiveTransform(q.astype(np.float32), rect)
        return hm / hm[2, 2], width, height

    hm, width, height = homography(quad)
    # A scan or a plain rotation is an affine map (no projective part): leave it
    # to the skew step, whose estimate is exact for a rotation; warp only when
    # the projective terms change the scale across the page by QUAD_MIN_PERSPECTIVE
    # — judged on the coarse corners, before snapping can add a fake residue.
    persp = abs(hm[2, 0]) * w + abs(hm[2, 1]) * h
    if persp < QUAD_MIN_PERSPECTIVE:
        return None
    quad = _snap_quad_to_grid_lines(gray, quad)
    hm, width, height = homography(quad)
    out_w, out_h = int(round(width)), int(round(height))
    if out_w < 50 or out_h < 50:
        return None
    _dewarp_by_quadrilateral.last_warp = (hm, (out_w, out_h))  # type: ignore[attr-defined]
    return cv2.warpPerspective(gray, hm, (out_w, out_h), borderValue=255)


# Grid-texture de-warp (D64): local grid frequencies per tile -> one homography.
GRID_TEXTURE_TILE_PX = 160        # tile side for the local 2-D spectrum
GRID_TEXTURE_STRIDE_PX = 80       # tile stride (50 % overlap)
GRID_TEXTURE_PITCH_PX = (4.0, 9.5)   # grid pitch band searched (1 mm at 4-9.5 px/mm)
GRID_TEXTURE_PEAK_SNR = 8.0       # spectral peak over the band's median: a tile shows grid
GRID_TEXTURE_MIN_TILES = 12       # fewer grid tiles: no texture fit
GRID_TEXTURE_MAX_RMS = 0.12       # fit residual (fraction of the pitch) above which the fit is rejected


def _local_grid_frequencies(gray: np.ndarray, tile: int = GRID_TEXTURE_TILE_PX,
                            stride: int = GRID_TEXTURE_STRIDE_PX) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Per tile (centre x, y): the grid's two frequency vectors (cycles/px), one
    near the x axis (vertical lines) and one near the y axis (horizontal lines),
    from the tile's high-passed 2-D spectrum with sub-bin (parabolic) peaks.
    Vignette and paper texture are low-frequency and leave after the high-pass;
    JPEG grain is broadband and does not move a sharp grid peak."""
    import cv2

    g = gray.astype(np.float32)
    h, w = g.shape
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    fy = np.fft.fftfreq(tile)[:, None]
    fx = np.fft.fftfreq(tile)[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    band = (fr >= 1.0 / GRID_TEXTURE_PITCH_PX[1]) & (fr <= 1.0 / GRID_TEXTURE_PITCH_PX[0])
    ang = np.degrees(np.arctan2(fy, fx))
    near_x = band & (np.abs(((ang + 90) % 180) - 90) <= 25)
    near_y = band & (np.abs(np.abs(ang) - 90) <= 25)

    def _sub(a: float, b: float, c: float) -> float:
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d

    out = []
    for y0 in range(0, h - tile + 1, stride):
        for x0 in range(0, w - tile + 1, stride):
            t = g[y0:y0 + tile, x0:x0 + tile]
            t = t - cv2.GaussianBlur(t, (0, 0), 6)
            spec = np.abs(np.fft.fft2(t * win))
            noise = float(np.median(spec[band])) + 1e-6
            found = []
            for k, mask in enumerate((near_x, near_y)):
                m = np.where(mask, spec, 0.0)
                iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
                if m[iy, ix] / noise < GRID_TEXTURE_PEAK_SNR:
                    found.append(None)
                    continue
                ly = [np.log(spec[(iy + d) % tile, ix] + 1e-9) for d in (-1, 0, 1)]
                lx = [np.log(spec[iy, (ix + d) % tile] + 1e-9) for d in (-1, 0, 1)]
                f = np.array([fx[0, ix] + _sub(*lx) / tile, fy[iy, 0] + _sub(*ly) / tile])
                # one half-plane per family, so that no reflection fits the data
                if (k == 0 and f[0] < 0) or (k == 1 and f[1] < 0):
                    f = -f
                found.append(f)
            if found[0] is not None and found[1] is not None:
                out.append((x0 + tile / 2.0, y0 + tile / 2.0, found[0], found[1]))
    return out


def _homography_jacobian(hm: np.ndarray, u: float, v: float) -> np.ndarray:
    w = hm[2, 0] * u + hm[2, 1] * v + 1.0
    x = hm[0, 0] * u + hm[0, 1] * v + hm[0, 2]
    y = hm[1, 0] * u + hm[1, 1] * v + hm[1, 2]
    return np.array([[(hm[0, 0] * w - x * hm[2, 0]) / w ** 2, (hm[0, 1] * w - x * hm[2, 1]) / w ** 2],
                     [(hm[1, 0] * w - y * hm[2, 0]) / w ** 2, (hm[1, 1] * w - y * hm[2, 1]) / w ** 2]])


def _fit_grid_homography(samples) -> tuple[np.ndarray, float, float]:
    """The homography H (rectified plane -> image, translation fixed at 0) under
    which the measured local grid frequencies are those of one square grid of
    pitch p: the frequency vectors transform with the inverse transpose of H's
    local Jacobian, so each tile gives two 2-vector residuals. Robust least
    squares from the identity; returns (H, p, rms residual in units of 1/p)."""
    from scipy.optimize import least_squares

    xs = np.array([s[0] for s in samples])
    ys = np.array([s[1] for s in samples])
    f1 = np.array([s[2] for s in samples])
    f2 = np.array([s[3] for s in samples])
    p = float(np.median(np.r_[1.0 / np.linalg.norm(f1, axis=1), 1.0 / np.linalg.norm(f2, axis=1)]))
    e1, e2 = np.array([1.0 / p, 0.0]), np.array([0.0, 1.0 / p])

    def _h(params):
        return np.array([[params[0], params[1], 0.0], [params[2], params[3], 0.0], [params[4], params[5], 1.0]])

    def residuals(params):
        hm = _h(params)
        hi = np.linalg.inv(hm)
        res = []
        for x, y, fa, fb in zip(xs, ys, f1, f2, strict=True):
            uvw = hi @ np.array([x, y, 1.0])
            jac = _homography_jacobian(hm, uvw[0] / uvw[2], uvw[1] / uvw[2])
            res.extend((jac.T @ fa - e1) * p)
            res.extend((jac.T @ fb - e2) * p)
        return np.array(res)

    sol = least_squares(residuals, np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), loss="soft_l1", f_scale=0.05, max_nfev=200)
    return _h(sol.x), p, float(np.sqrt(np.mean(sol.fun ** 2)))


def _dewarp_by_grid_texture(gray: np.ndarray) -> np.ndarray | None:
    """Rectify a photographed page from its grid texture instead of its edges (D64).

    The edge-based quadrilateral breaks when a shadow edge, a trace or JPEG
    grain wins a Hough vote (2 % photo: r 0.65, a third of the pages under 0.5;
    rotated vignetted 3.5 %: r 0.39). The grid itself is a periodic texture that
    survives vignette, grain and JPEG, and a perspective changes its local
    frequency smoothly across the page — a homography fitted to the per-tile
    frequencies rectifies it. Returns None without enough grid tiles, with a poor
    fit, or when the fit is affine (a rotation or a scan: the de-skew step is
    exact there and this estimate is not)."""
    import cv2

    samples = _local_grid_frequencies(gray)
    if len(samples) < GRID_TEXTURE_MIN_TILES:
        return None
    hm, _p, rms = _fit_grid_homography(samples)
    if rms > GRID_TEXTURE_MAX_RMS:
        return None
    h, w = gray.shape
    if abs(hm[2, 0]) * w + abs(hm[2, 1]) * h < QUAD_MIN_PERSPECTIVE:
        return None
    hi = np.linalg.inv(hm)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ hi.T
    corners = corners[:, :2] / corners[:, 2:3]
    mn, mx = corners.min(axis=0), corners.max(axis=0)
    out_w, out_h = int(round(mx[0] - mn[0])), int(round(mx[1] - mn[1]))
    if not (50 < out_w < 4 * w and 50 < out_h < 4 * h):
        return None
    shift = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]])
    _dewarp_by_grid_texture.last_warp = (shift @ hi, (out_w, out_h))  # type: ignore[attr-defined]
    return cv2.warpPerspective(gray, shift @ hi, (out_w, out_h), borderValue=255)


def _compose_warps(steps: list[tuple[np.ndarray, tuple[int, int]]], size: tuple[int, int]):
    """One 3x3 map and output size equivalent to applying ``steps`` in order."""
    total = np.eye(3)
    for m, s in steps:
        total = np.asarray(m, dtype=float) @ total
        size = s
    return total, size


def apply_dewarp(img: np.ndarray, transform=None, *, nearest: bool = False, border: int = 0) -> np.ndarray:
    """Warp a companion image (an ink mask, the colour photo) exactly like the last
    ``dewarp_page`` call warped its grayscale input — ``dewarp_page.last_transform``."""
    import cv2

    hm, size = transform or getattr(dewarp_page, "last_transform", (np.eye(3), img.shape[1::-1]))
    if np.allclose(hm, np.eye(3)) and tuple(size) == tuple(img.shape[1::-1]):
        return img
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    src = img.astype(np.uint8) if img.dtype == bool else img
    out = cv2.warpPerspective(src, hm, tuple(int(v) for v in size), flags=flags, borderValue=border)
    return out.astype(bool) if img.dtype == bool else out


def dewarp_page(gray: np.ndarray) -> np.ndarray:
    """Undo a perspective. First choice (D64): the homography fitted to the grid
    texture — on 100 rendered pages it reads the 2 % photo at r 0.81 (edges:
    0.65), the rotated vignetted 3.5 % page at 0.84 (0.39) and pure 5 %
    perspective at 0.91 (0.84); it steps aside for affine cases (scans), where
    the de-skew is exact. Second choice: the page quadrilateral (the
    grid-textured region's hull, sides snapped to the outermost grid lines)
    mapped to a rectangle, then the vanishing-point step for the few-pixel
    residual. Without either, the vanishing-point method alone handles a mild
    perspective (to ~2 %). The input is returned unchanged when none applies.

    The composed map of the path taken is left in ``dewarp_page.last_transform``
    (3x3 matrix, output size) so ``apply_dewarp`` can warp companion images."""
    h, w = gray.shape
    steps: list[tuple[np.ndarray, tuple[int, int]]] = []

    def finish(out: np.ndarray) -> np.ndarray:
        dewarp_page.last_transform = _compose_warps(steps, (w, h))  # type: ignore[attr-defined]
        return out

    def refine(img: np.ndarray) -> np.ndarray:
        out = _refine_affine(img)
        steps.extend(getattr(_refine_affine, "last_warps", []))
        return out

    by_texture = _dewarp_by_grid_texture(gray)
    if by_texture is not None:
        steps.append(_dewarp_by_grid_texture.last_warp)  # type: ignore[attr-defined]
        return finish(refine(by_texture))
    by_quad = _dewarp_by_quadrilateral(gray)
    if by_quad is not None:
        # Sanity: the rectified page must show a square grid (pitch along x and y
        # within 10 %); a wrong quadrilateral yields none or a skewed one.
        px, py = estimate_grid_pitch_xy(by_quad)
        if px and py and 0.9 < px / py < 1.1:
            steps.append(_dewarp_by_quadrilateral.last_warp)  # type: ignore[attr-defined]
            second = _dewarp_by_vanishing_points(by_quad)
            if second is not None:
                px2, py2 = estimate_grid_pitch_xy(second)
                if not (px2 and py2 and 0.95 < px2 / py2 < 1.05
                        and _line_angle_spread(second) < 0.8 * _line_angle_spread(by_quad)):
                    second = None
            if second is not None:
                steps.append(_dewarp_by_vanishing_points.last_warp)  # type: ignore[attr-defined]
            return finish(refine(second if second is not None else by_quad))
    vp = _dewarp_by_vanishing_points(gray)
    if vp is not None:
        steps.append(_dewarp_by_vanishing_points.last_warp)  # type: ignore[attr-defined]
        return finish(refine(vp))
    return finish(gray)


def _dewarp_by_vanishing_points(gray: np.ndarray) -> np.ndarray | None:
    """The grid's horizontal and vertical line families converge to vanishing
    points under perspective; the homography that sends their vanishing line
    back to infinity rectifies the page. Returns None when the families are
    parallel already or the fit does not make them more parallel (self-check)."""
    import cv2

    h, w = gray.shape
    segs = _grid_segments(gray)
    if segs is None:
        return None
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    horiz = segs[np.abs(ang) < 10]
    vert = segs[np.abs(np.abs(ang) - 90) < 10]
    vh = _vanishing_point(horiz, True)
    vv = _vanishing_point(vert, False)
    if vh is None and vv is None:
        return None
    if vh is None:
        vh = np.array([1.0, 0.0, 0.0])
    if vv is None:
        vv = np.array([0.0, 1.0, 0.0])
    linf = np.cross(vh, vv)
    if abs(linf[2]) < 1e-12:
        return None
    linf = linf / linf[2]
    hm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [linf[0], linf[1], 1.0]])
    # Keep the page centre fixed and its scale ~unchanged: normalise by the centre's w.
    c = hm @ np.array([w / 2, h / 2, 1.0])
    if not np.isfinite(c).all() or abs(c[2]) < 1e-6:
        return None
    scale = np.array([[1 / c[2], 0, 0], [0, 1 / c[2], 0], [0, 0, 1]])
    hm = scale @ hm
    # Re-centre so the page centre stays put.
    c2 = hm @ np.array([w / 2, h / 2, 1.0]); c2 = c2 / c2[2]
    shift = np.array([[1, 0, w / 2 - c2[0]], [0, 1, h / 2 - c2[1]], [0, 0, 1]])
    hm = shift @ hm
    warped = cv2.warpPerspective(gray, hm, (w, h), borderValue=255)
    # Self-check: keep the rectified page only if its grid lines became clearly
    # more parallel (a rotation or a noisy fit must not trigger a warp).
    if not _line_angle_spread(warped) < _line_angle_spread(gray):
        return None
    _dewarp_by_vanishing_points.last_warp = (hm, (w, h))  # type: ignore[attr-defined]
    return warped


def _refine_affine(warped: np.ndarray) -> np.ndarray:
    """After a projective step an affine residue remains (shear, anisotropic
    scale): make the horizontal family horizontal and the vertical family vertical,
    then fix the aspect from the grid squares (pitch along x vs y)."""
    import cv2

    h, w = warped.shape
    steps: list[tuple[np.ndarray, tuple[int, int]]] = []
    segs2 = _grid_segments(warped)
    if segs2 is not None and len(segs2) >= 8:
        a2 = np.arctan2(segs2[:, 3] - segs2[:, 1], segs2[:, 2] - segs2[:, 0])
        hh = a2[np.abs(np.degrees(a2)) < 10]
        vv = a2[np.abs(np.abs(np.degrees(a2)) - 90) < 10]
        if len(hh) >= 4 and len(vv) >= 4:
            th = float(np.median(hh))
            vv = np.where(vv < 0, vv + np.pi, vv)          # vertical direction, pointing down
            tv = float(np.median(vv))
            basis = np.array([[np.cos(th), np.cos(tv)], [np.sin(th), np.sin(tv)]])
            if abs(np.linalg.det(basis)) > 0.5:
                a_inv = np.linalg.inv(basis)
                c = np.array([w / 2, h / 2])
                m = np.hstack([a_inv, (c - a_inv @ c).reshape(2, 1)])
                warped = cv2.warpAffine(warped, m, (w, h), borderValue=255)
                steps.append((np.vstack([m, [0.0, 0.0, 1.0]]), (w, h)))
    px, py = estimate_grid_pitch_xy(warped)
    if px and py and 0.8 < px / py < 1.25 and abs(px / py - 1.0) > 0.01:
        m = np.array([[1.0, 0.0, 0.0], [0.0, px / py, h / 2 * (1 - px / py)]])
        warped = cv2.warpAffine(warped, m, (w, h), borderValue=255)
        steps.append((np.vstack([m, [0.0, 0.0, 1.0]]), (w, h)))
    _refine_affine.last_warps = steps  # type: ignore[attr-defined]
    return warped


def digitize_page(
    path: str | Path, *, px_per_mm: float | None = None, mm_per_s: float = 25.0,
    mm_per_mv: float | None = None, dewarp: bool = True, tracked: bool = True,
) -> dict[str, np.ndarray]:
    """Recover each lead's µV trace (at px_per_s Hz) from a 3x4 + rhythm-strip page.

    ``px_per_mm`` None -> estimated from the grid pitch (fallback 6); ``mm_per_mv``
    None -> read from the calibration pulse when present (fallback 10). Returns
    ``{lead: trace_uv}`` (12 panel leads at 2.5 s, plus ``"II_rhythm"`` at 10 s);
    ``digitize_page.last_coverage`` / ``last_calibration`` hold the provenance.
    ``tracked`` (D47): skeletonise the ink and follow each panel's own trace by
    continuity instead of taking the median of all ink per column; ``False`` is
    the pre-D47 behaviour, kept for comparison.
    """
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if dewarp:
        gray = dewarp_page(gray)
    skew = estimate_skew(gray)
    if abs(skew) > 0.2:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    calib: dict = {"px_per_mm_given": px_per_mm, "mm_per_mv_given": mm_per_mv}
    if px_per_mm is None:
        est = estimate_grid_pitch(gray)
        px_per_mm = est if est is not None else 6.0
        calib["px_per_mm_estimated"] = None if est is None else round(est, 2)
    def layout(ppm: float):
        ppm_i = max(1, int(round(ppm)))
        geo = page_geometry(ppm_i, mm_per_s, mm_per_mv or 10.0)
        # One component-filtered ink mask for everything: JPEG ringing, salt noise
        # and lead labels are short blobs; a single stray pixel at the page edge
        # would otherwise stretch the row/column extents to the whole image.
        mk = _keep_long_components(trace_mask(gray, ppm_i), int(MIN_COMPONENT_WIDTH * geo.panel_w))
        # Baselines from the horizontal ink projection (ink piles up on the flat
        # segments), panel boxes from the layout geometry around them: a band cut
        # at a profile threshold would clip the QRS/T excursions. Fall back to geometry.
        bd = _bands(mk.sum(axis=1).astype(float), 4, min_gap=max(int(3 * ppm_i), int(0.02 * gray.shape[0])))
        return ppm_i, geo, mk, bd

    px_per_mm_i, g, mask, bands = layout(px_per_mm)
    if len(bands) != 4 and calib.get("px_per_mm_given") is None:
        for alt in (px_per_mm * 2, px_per_mm / 2, 6.0):
            ppm_i2, g2, mask2, bands2 = layout(alt)
            if len(bands2) == 4:
                px_per_mm, px_per_mm_i, g, mask, bands = alt, ppm_i2, g2, mask2, bands2
                break
    if len(bands) == 4 and calib.get("px_per_mm_given") is None:
        # The layout itself is a second scale reference: rows are ROW_HEIGHT_MM
        # apart. When the grid estimate disagrees (blur leaves only the 5-mm lines
        # and P vs 2P is ambiguous), take the grid harmonic nearest the row pitch,
        # or the row pitch itself.
        centres = np.array([(a + b) / 2 for a, b in bands])
        ppm_rows = float(np.median(np.diff(centres))) / ROW_HEIGHT_MM
        calib["px_per_mm_from_rows"] = round(ppm_rows, 2)
        # The row pitch is the primary scale (it is what the panel geometry is
        # built from); the grid estimate refines it only when the two agree.
        new_ppm = (px_per_mm + ppm_rows) / 2 if abs(px_per_mm - ppm_rows) <= 0.05 * ppm_rows else ppm_rows
        if abs(new_ppm - px_per_mm) > 1e-6:
            px_per_mm = new_ppm
            calib["px_per_mm_estimated"] = round(px_per_mm, 2)
            px_per_mm_i, g, mask, bands = layout(px_per_mm)
    half = g.row_h // 2
    if len(bands) == 4:
        rows = [(max(0, (a + b) // 2 - half), min(gray.shape[0], (a + b) // 2 + half)) for a, b in bands]
    else:
        rows = [(g.panel_box(r, 0)[2], g.panel_box(r, 0)[3]) for r in range(4)]
    # Column extent from the ink of the middle row, split into four equal panels.
    ink_x = np.flatnonzero(mask[rows[1][0]:rows[1][1]].any(axis=0))
    if ink_x.size > 4 * px_per_mm_i:
        xa, xb = int(ink_x[0]), int(ink_x[-1]) + 1
    else:
        xa, xb = g.margin, g.margin + 4 * g.panel_w
    # Calibration pulse at the left of the rows: 1 mV -> px, and the traces start after it.
    pulse = detect_calibration_pulse(mask, xa, rows[1][0], rows[1][1], px_per_mm)
    if pulse is not None:
        height_px, right = pulse
        if mm_per_mv is None:
            mm_per_mv = height_px / px_per_mm / PULSE_HEIGHT_MV
        calib["mm_per_mv_from_pulse"] = round(height_px / px_per_mm, 2)
        xa = right + int(1.5 * px_per_mm)
        ink_x = np.flatnonzero(mask[rows[1][0]:rows[1][1], xa:].any(axis=0))
        if ink_x.size:
            xa = xa + int(ink_x[0])
    mm_per_mv = mm_per_mv or 10.0
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    calib.update(px_per_mm=round(float(px_per_mm), 2), mm_per_mv=round(float(mm_per_mv), 2), pulse_found=pulse is not None)
    edges = np.linspace(xa, xb, 5).astype(int)
    cols = [(int(edges[c]), int(edges[c + 1])) for c in range(4)]
    out: dict[str, np.ndarray] = {}
    coverages = []
    thin = thin_mask(mask) if tracked else None

    def rows_of(x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        if thin is None:
            return _trace_rows(mask, x0, x1, y0, y1)
        return _trace_rows_tracked(mask, thin, x0, x1, y0, y1)

    for r, (y0, y1) in enumerate(rows[:3]):
        for c, (x0, x1) in enumerate(cols):
            lead = LAYOUT_3X4[r][c]
            trace, cov = _fill_gaps(rows_of(x0, x1, y0, y1))
            base = float(np.median(trace))                  # panel baseline = median row
            out[lead] = ((base - trace) * uv_per_px).astype(np.float32)
            coverages.append(cov)
    y0, y1 = rows[3]
    trace, cov = _fill_gaps(rows_of(xa, xb, y0, y1))
    out["II_rhythm"] = ((float(np.median(trace)) - trace) * uv_per_px).astype(np.float32)
    coverages.append(cov)
    digitize_page.last_coverage = float(np.mean(coverages))  # type: ignore[attr-defined]
    digitize_page.last_calibration = calib  # type: ignore[attr-defined]
    return out


def page_to_signal(
    traces: dict[str, np.ndarray], *, px_per_mm: float | None = None, mm_per_s: float = 25.0,
    mm_per_mv: float | None = None, coverage: float | None = None, calibration: dict | None = None,
) -> SignalRepresentation:
    """10-s 12-lead SignalRepresentation from page traces, with lead time windows."""
    calibration = calibration or getattr(digitize_page, "last_calibration", None) or {}
    px = px_per_mm if px_per_mm is not None else calibration.get("px_per_mm", 6.0)
    g = page_geometry(max(1, int(round(px))), mm_per_s, mm_per_mv or calibration.get("mm_per_mv", 10.0))
    fs = float(g.px_per_s)
    n = int(round(4 * PANEL_SECONDS * fs))
    seg = int(round(PANEL_SECONDS * fs))
    signal = np.zeros((12, n), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    for row, leads in enumerate(LAYOUT_3X4):
        for col, lead in enumerate(leads):
            li = STANDARD_12_LEADS.index(lead)
            if lead == RHYTHM_LEAD and "II_rhythm" in traces:
                t = traces["II_rhythm"][:n]
                signal[li, : len(t)] = t
                windows[lead] = (0.0, 4 * PANEL_SECONDS)
                continue
            t = traces.get(lead)
            if t is None:
                continue
            t = t[:seg]
            signal[li, col * seg: col * seg + len(t)] = t
            windows[lead] = (col * PANEL_SECONDS, (col + 1) * PANEL_SECONDS)
    cov = coverage if coverage is not None else getattr(digitize_page, "last_coverage", None)
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS),
        duration_s=float(n / fs), source="image",
        provenance=Provenance(digitized=True, confidence=None if cov is None else round(cov, 3),
                              lead_windows=windows,
                              notes=["3x4 page: each lead valid only inside its time window"]
                                    + ([f"calibration: {calibration}"] if calibration else [])),
    )


def digitize_page_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """Convenience: ``digitize_page`` + ``page_to_signal`` with shared calibration."""
    traces = digitize_page(path, **kw)
    kw.pop("dewarp", None)
    kw.pop("tracked", None)
    return page_to_signal(traces, coverage=getattr(digitize_page, "last_coverage", None),
                          calibration=getattr(digitize_page, "last_calibration", None), **kw)
