"""Phone photos of an ECG device's own screen (D120).

Portable recorders (TeleECG and the like) show a preview of N leads stacked as
rows in one time window — a "6x2" display pages between the six limb and the six
chest leads. A photo of that screen is not paper: the grid is grey, not red, the
panel is a bright rectangle inside a dark bezel, there is no calibration pulse,
and the window is short (~2 s, three beats). The paper digitisers (page 3x4,
strip Nx1) look for a red grid and find nothing.

1. the panel is the largest bright, unsaturated region; it ends where the rows
   stop being bright across the whole width (the device's info box below);
2. ink is dark relative to the local background — the grey grid stays lighter
   (ratio ~0.9) than even a faint trace (0.55-0.8);
3. row baselines are the peaks of the horizontal ink projection; each row is
   straightened and read with the page tracker (D47), which keeps a tall R of
   the row below out of the row above;
4. the scale is **assumed**: the grid's major pitch is 5 mm, 25 mm/s, 10 mm/mV
   (the display usually says so; the photo does not). ``provenance.notes`` says it;
5. six rows without a lead order: limb (I..aVF) when the first three satisfy
   Einthoven (II = I + III), chest (V1..V6) otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation

SCREEN_BRIGHT_V = 165          # HSV value above this and ...
SCREEN_BRIGHT_MAX_S = 60       # ... saturation below this = the lit panel
SCREEN_ROW_FULL = 0.6          # a panel row is bright across this much of the width
SCREEN_FOOTER_ROWS = 5         # this many non-full rows in a row below 30 % of the height end the panel
SCREEN_INK_RATIO = 0.83        # darker than this fraction of the local background = trace
SCREEN_BG_KERNEL = 31
SCREEN_LABEL_FRACTION = 0.22   # the lead labels and the sweep's leftovers sit in the left margin
SCREEN_RIGHT_MARGIN_PX = 12
SCREEN_GRID_MM = 5.0           # assumed major grid pitch
SCREEN_GRID_PERIOD_PX = (15.0, 150.0)
SCREEN_FUNDAMENTAL_FRACTION = 0.5
SCREEN_MIN_PANEL_FRACTION = 0.03
SCREEN_MAX_PANEL_FRACTION = 0.8   # a panel filling the photo is a paper page, not a screen
SCREEN_MAX_RED_FRACTION = 0.02    # red-grid pixels inside the panel: more = paper
SCREEN_LIMB_EINTHOVEN_R = 0.9
SCREEN_LIGHT_MIN_V = 170          # a lit panel's median brightness
SCREEN_RIM_CONTRAST = 80          # panel minus its frame (see _rim_contrast)
SCREEN_LIGHT_MIN_INK = 0.005      # a lit panel with less trace ink is a button or an info box
SCREEN_DARK_INK_RATIO = 1.35      # dark screens: brighter than this x the local background ...
SCREEN_DARK_TRACE_RATIO = 1.6     # ... and for reading the trace (the grid lines reach ~1.3)
SCREEN_DARK_MIN_S = 70            # ... coloured (white text and glare are not) ...
SCREEN_DARK_MIN_V = 110
SCREEN_DARK_HUE = (20, 95)        # ... green to yellow (the blue header bar is not a trace)
SCREEN_DARK_MIN_DENSITY = 0.02
SCREEN_DARK_MAX_BG_V = 185
SCREEN_DARK_CLUSTER_FRACTION = 0.1   # ink clusters this large (of the largest) belong to the panel
SCREEN_TWO_COLUMN_ASPECT = 1.45   # panel width / height above this = a 6x2 display
SCREEN_6X2_COLUMN_S = 5.0         # each column of a 6x2 display: 5 s, one after the other
SCREEN_COLUMN_LABEL_FRACTION = 0.08
SCREEN_BOX_PITCHES = 0.75        # tracker box around a baseline, in row pitches each way
SCREEN_SPAN_FRACTION = 0.3        # a column's traces run where long-stroke ink reaches this of its median
SCREEN_PORTRAIT_RATIO = 1.3       # panel height / width above this = turned photo
SCREEN_STROKE_KERNEL = 0.06       # ... if long vertical strokes then outweigh horizontal ones
SCREEN_BAR_MIN_FRACTION = 0.03    # the blue button bar of a dark recorder, of the photo
SCREEN_UPRIGHT_MIN = 0.35         # fewer upright QRS than this: the photo is upside down
LIMB_6 = ["I", "II", "III", "aVR", "aVL", "aVF"]
CHEST_6 = ["V1", "V2", "V3", "V4", "V5", "V6"]


def find_screen_panel(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the lit waveform panel, or None."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 2] > SCREEN_BRIGHT_V) & (hsv[..., 1] < SCREEN_BRIGHT_MAX_S)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    area = bgr.shape[0] * bgr.shape[1]
    order = [k for k in np.argsort(-st[1:, 4]) + 1
             if SCREEN_MIN_PANEL_FRACTION * area <= st[k, 2] * st[k, 3] <= SCREEN_MAX_PANEL_FRACTION * area]
    if not order:
        return None
    k = int(order[0])
    x, y, w, h = (int(v) for v in st[k, :4])
    full = (lab[y:y + h, x:x + w] == k).mean(axis=1) > SCREEN_ROW_FULL
    bottom = h
    bad = np.flatnonzero(~full)
    for run in (np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if bad.size else []):
        if len(run) >= SCREEN_FOOTER_ROWS and run[0] > 0.3 * h:
            bottom = int(run[0])
            break
    return x, y, x + w, y + bottom


def _rim_contrast(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Panel brightness minus the dark quarter of a thin ring around it: a lit screen sits
    in a black frame (TeleECG 169, synthetic 198); paper on a table does not (19-33)."""
    import cv2

    x0, y0, x1, y1 = box
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    d = max(4, int(0.03 * max(x1 - x0, y1 - y0)))
    ring = np.concatenate([v[max(0, y0 - d):y0, x0:x1].ravel(), v[y1:y1 + d, x0:x1].ravel(),
                           v[y0:y1, max(0, x0 - d):x0].ravel(), v[y0:y1, x1:x1 + d].ravel()])
    if ring.size == 0:
        return 0.0
    return float(np.median(v[y0:y1, x0:x1])) - float(np.percentile(ring, 25))


def find_dark_panel(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of a dark screen with bright coloured traces (green/yellow on
    black, D121), from the densest cluster of such ink; None if there is none."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, SCREEN_BG_KERNEL).astype(np.float32) + 1.0
    hue = hsv[..., 0]
    ink = ((g / bg > SCREEN_DARK_INK_RATIO) & (hsv[..., 1] > SCREEN_DARK_MIN_S) & (hsv[..., 2] > SCREEN_DARK_MIN_V)
           & (hue >= SCREEN_DARK_HUE[0]) & (hue <= SCREEN_DARK_HUE[1])).astype(np.uint8)
    k = max(9, int(0.03 * max(bgr.shape[:2])))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((k, k), np.uint8)))
    if n < 2:
        return None
    # Rows far apart (a tall display) stay separate clusters: the panel spans all of the
    # large ones.
    big = [j for j in range(1, n) if st[j, 4] >= SCREEN_DARK_CLUSTER_FRACTION * st[1:, 4].max()]
    x, y = int(min(st[j, 0] for j in big)), int(min(st[j, 1] for j in big))
    w = int(max(st[j, 0] + st[j, 2] for j in big)) - x
    h = int(max(st[j, 1] + st[j, 3] for j in big)) - y
    area = bgr.shape[0] * bgr.shape[1]
    if not SCREEN_MIN_PANEL_FRACTION * area <= w * h <= SCREEN_MAX_PANEL_FRACTION * area:
        return None
    if ink[y:y + h, x:x + w].mean() < SCREEN_DARK_MIN_DENSITY or np.median(hsv[y:y + h, x:x + w, 2]) > SCREEN_DARK_MAX_BG_V:
        return None
    return x, y, x + w, y + h


def screen_view(bgr: np.ndarray) -> tuple[str, np.ndarray, tuple[int, int, int, int]] | None:
    """(mode, grey panel, box) for a photo of a device screen — "light" (dark trace on a lit
    panel) or "dark" (bright trace on black); None for anything else (paper)."""
    import cv2

    box = find_screen_panel(bgr)
    if box is not None:
        x0, y0, x1, y1 = box
        p = bgr[y0:y1, x0:x1].astype(int)
        red = (p[..., 2] - p[..., 0] > 40) & (p[..., 2] > 150)
        v = float(np.median(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2]))
        gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        # A white button or info box is lit and framed too, but carries no trace.
        if (red.mean() <= SCREEN_MAX_RED_FRACTION and v >= SCREEN_LIGHT_MIN_V
                and _rim_contrast(bgr, box) >= SCREEN_RIM_CONTRAST
                and _ink(gray)[1].mean() >= SCREEN_LIGHT_MIN_INK):
            return "light", gray, box
    box = find_dark_panel(bgr)
    if box is not None:
        x0, y0, x1, y1 = box
        return "dark", cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), box
    return None


def is_device_screen(bgr: np.ndarray) -> bool:
    """A photo of a device's own display (light or dark), not paper."""
    return screen_view(bgr) is not None


def _major_pitch_px(g: np.ndarray, ink: np.ndarray) -> float | None:
    """Major grid period (px) along x: the strongest spectral line of the column profile."""
    prof = np.where(ink, 1.0, g).mean(axis=0)
    prof = prof - prof.mean()
    if prof.size < 64:
        return None
    f = np.abs(np.fft.rfft(prof * np.hanning(prof.size)))
    fr = np.fft.rfftfreq(prof.size)
    # A sharp 1-px grid has strong harmonics: only periods a 5-mm square can have at
    # phone resolution (3-30 px/mm) count.
    ok = (fr >= 1.0 / SCREEN_GRID_PERIOD_PX[1]) & (fr <= 1.0 / SCREEN_GRID_PERIOD_PX[0])
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    loc = [i for i in idx if f[i] >= f[i - 1] and f[i] >= f[min(i + 1, f.size - 1)]]
    if not loc:
        return None
    top_i = max(loc, key=lambda i: f[i])
    # The fundamental, not a harmonic: the longest period among the strongest line's
    # sub-harmonics (2x, 3x, 4x its period) that still carries half its power. Any lower
    # line is not the grid (a narrow panel has labels and a few beats at low frequency).
    best = top_i
    for m in (2, 3, 4):
        j = int(round(top_i / m))
        near = [i for i in loc if abs(i - j) <= 1 and fr[i] >= 1.0 / SCREEN_GRID_PERIOD_PX[1]]
        if near and f[max(near, key=lambda i: f[i])] >= SCREEN_FUNDAMENTAL_FRACTION * f[top_i]:
            best = max(near, key=lambda i: f[i])
    return float(1.0 / fr[best])


def _ink(gray: np.ndarray, mode: str = "light") -> tuple[np.ndarray, np.ndarray]:
    """(brightness / local background, trace mask). Light screens: the trace is darker than
    0.83 of its background, the grey grid is not. Dark screens: the grid lines are brighter
    than the background too, so the trace must be 1.6 x brighter (text stays; the row
    search below keeps only long horizontal strokes)."""
    import cv2

    bg = cv2.medianBlur(gray, SCREEN_BG_KERNEL).astype(np.float32) + 1.0
    ratio = gray.astype(np.float32) / bg
    if mode == "dark":
        return 1.0 / np.maximum(ratio, 1e-3), ratio > SCREEN_DARK_TRACE_RATIO
    return ratio, ratio < SCREEN_INK_RATIO


def _rows_horizontal(mask: np.ndarray) -> bool:
    """False for a photo taken with the screen turned by 90 degrees: the panel is clearly
    taller than wide *and* the baselines run vertically (more long vertical than long
    horizontal strokes). Neither alone: TeleECG is a portrait device with a ~1:1 preview,
    a 1.8-s window renders taller than wide, and QRS strokes of overlapping rows join into
    vertical lines across a landscape panel."""
    import cv2

    h, w = mask.shape
    if h <= SCREEN_PORTRAIT_RATIO * w:
        return True
    m = mask.astype(np.uint8)
    k = max(5, int(SCREEN_STROKE_KERNEL * max(h, w)))
    horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, k), np.uint8)).sum()
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8)).sum()
    return bool(horiz >= vert)


def _bar_side(bgr: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    """Where the device's blue button bar sits relative to the panel ("left", "right",
    "below", "above"), or None. On the dark 6x2 recorders it runs along the bottom of the
    display, so a bar beside the traces means the photo was taken turned by 90 degrees —
    and says which way (the ink cannot: on a turned photo the QRS strokes of all rows
    join into lines along the time axis)."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    if n < 2:
        return None
    j = 1 + int(np.argmax(st[1:, 4]))
    if st[j, 4] < SCREEN_BAR_MIN_FRACTION * bgr.shape[0] * bgr.shape[1]:
        return None
    x0, y0, x1, y1 = box
    dx, dy = cen[j][0] - (x0 + x1) / 2.0, cen[j][1] - (y0 + y1) / 2.0
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "below" if dy > 0 else "above"


def _long_strokes(mask: np.ndarray) -> np.ndarray:
    """Only ink in horizontal runs of >= 3 % of the width: baselines and slow waves stay,
    text, grid and QRS strokes go."""
    import cv2

    k = max(5, int(0.03 * mask.shape[1]))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, k), np.uint8)).astype(bool)


def _columns(mask: np.ndarray) -> list[tuple[int, int]]:
    """One column, or two side by side on a wide display (6x2). The ink does not show the
    seam — the right column starts where the left one ends and QRS strokes of all rows
    leave the same gaps — so the display's shape decides (wide 6x2 screens 1.65-1.9,
    a 6x1 preview ~1.05), and the cut is the thinnest long-stroke ink in the middle fifth."""
    h, w = mask.shape
    if w < SCREEN_TWO_COLUMN_ASPECT * h:
        return [(0, w)]
    dens = _long_strokes(mask).sum(axis=0).astype(float)
    k = max(3, int(0.01 * w))
    sm = np.convolve(dens, np.ones(k) / k, mode="same")
    a, b = int(0.4 * w), int(0.6 * w)
    cut = a + int(np.argmin(sm[a:b]))
    return [(0, cut), (cut, w)]


def _trace_span(mask: np.ndarray, c0: int, c1: int) -> tuple[int, int]:
    """Where a column's traces run: long horizontal strokes (labels, the sweep's leftover
    hooks and text are short) at >= 30 % of their median density across x."""
    dens = _long_strokes(mask[:, c0:c1]).sum(axis=0).astype(float)
    k = max(3, int(0.01 * (c1 - c0)))
    sm = np.convolve(dens, np.ones(k) / k, mode="same")
    ref = float(np.median(sm[sm > 0])) if (sm > 0).any() else 0.0
    on = np.flatnonzero(sm >= SCREEN_SPAN_FRACTION * ref) if ref > 0 else np.array([], int)
    if on.size < 2:
        return c0 + int(SCREEN_LABEL_FRACTION * (c1 - c0)), c1 - SCREEN_RIGHT_MARGIN_PX
    # Never onto the panel's own edge: the frame is a full column of "ink".
    return (max(c0 + SCREEN_RIGHT_MARGIN_PX, c0 + int(on[0])),
            min(c1 - SCREEN_RIGHT_MARGIN_PX, c0 + int(on[-1]) + 1))


def _read_rows(mask: np.ndarray, x_left: int, x_right: int, n_rows: int | None,
               uv_per_px: float) -> tuple[list[np.ndarray], list[float]]:
    from scipy.signal import find_peaks

    from cardiolens.layer1_ingestion.digitize import _fill_gaps, _trace_rows_tracked, thin_mask
    from cardiolens.layer1_ingestion.strip import straighten

    prof = _long_strokes(mask[:, x_left:x_right]).sum(axis=1).astype(float)
    guess = n_rows or 6
    pk, _ = find_peaks(prof, distance=max(3, int(0.6 * mask.shape[0] / guess)),
                       height=0.05 * (x_right - x_left))
    if n_rows is None:
        n_rows = len(pk) if len(pk) in (3, 6, 12) else (6 if len(pk) >= 6 else len(pk))
    if len(pk) < n_rows or n_rows < 3:
        raise ValueError(f"screen: expected {n_rows} rows, found {len(pk)}")
    base = np.sort(pk[np.argsort(-prof[pk])[:n_rows]])
    row_pitch = float(np.median(np.diff(base)))
    half = int(1.5 * row_pitch)
    t0, t1 = half - int(SCREEN_BOX_PITCHES * row_pitch), half + int(SCREEN_BOX_PITCHES * row_pitch)
    rows, coverages = [], []
    for b in base:
        band = straighten(mask, np.full(mask.shape[1], float(b)), half)
        raw = _trace_rows_tracked(band, thin_mask(band), x_left, x_right, t0, t1, baseline_weight=1.0)
        trace, cov = _fill_gaps(raw)
        rows.append(((float(np.median(trace)) - trace) * uv_per_px).astype(np.float32))
        coverages.append(cov)
    return rows, coverages


def _upright_fraction(rows: list[np.ndarray]) -> float:
    """Share of rows whose largest deflection points up — most standard leads (I, II, aVF,
    V3-V6) have an upright QRS; an upside-down photo turns that over."""
    up = [float(np.max(r) - np.median(r)) > float(np.median(r) - np.min(r)) for r in rows if r.size]
    return float(np.mean(up)) if up else 1.0


def digitize_screen(path: str | Path, *, lead_order: list[str] | None = None,
                    mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                    grid_mm: float = SCREEN_GRID_MM) -> dict[str, np.ndarray]:
    """``{lead: trace_uv}`` for a photo of a device screen — light or dark (D121), one column
    or two side by side, upright, upside down or turned by 90 degrees;
    ``digitize_screen.last_calibration`` / ``last_coverage`` hold the provenance.
    Raises ValueError when no screen or rows are found."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    view = screen_view(bgr)
    if view is None:
        raise ValueError("screen: no device screen found")
    mode, gray, box = view
    ratio, mask = _ink(gray, mode)
    turns = 0
    side = _bar_side(bgr, box) if mode == "dark" else None
    if side in ("left", "right"):          # the display's bottom points that way
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE if side == "left" else cv2.ROTATE_90_CLOCKWISE)
        turns = 3 if side == "left" else 1
        ratio, mask = _ink(gray, mode)
    elif mode == "dark" and side is None and not _rows_horizontal(mask):
        gray, turns = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 1
        ratio, mask = _ink(gray, mode)
    # A lit preview (TeleECG) is read as photographed; turned only if no rows are found
    # that way and the panel is taller than wide (D122: the stroke test turned a narrow
    # 1.8-s window).
    if mode == "light" and turns == 0:
        try:
            return _digitize_view(gray, mode, box, 0, lead_order, mm_per_s, mm_per_mv, grid_mm)
        except ValueError:
            if gray.shape[0] <= SCREEN_PORTRAIT_RATIO * gray.shape[1]:
                raise
            gray, turns = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 1
    return _digitize_view(gray, mode, box, turns, lead_order, mm_per_s, mm_per_mv, grid_mm)


def _digitize_view(gray: np.ndarray, mode: str, box: tuple[int, int, int, int], turns: int,
                   lead_order: list[str] | None, mm_per_s: float, mm_per_mv: float,
                   grid_mm: float) -> dict[str, np.ndarray]:
    ratio, mask = _ink(gray, mode)
    cols = _columns(mask)

    def spans_of(m: np.ndarray, cs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # A lit single-column preview (TeleECG) keeps its labels and the sweep's leftover
        # hooks in a fixed left margin; a tight dark panel or a 6x2 display has its traces
        # found from the ink.
        if mode == "light" and len(cs) == 1:
            c0, c1 = cs[0]
            return [(c0 + int(SCREEN_LABEL_FRACTION * (c1 - c0)), c1 - SCREEN_RIGHT_MARGIN_PX)]
        return [_trace_span(m, c0, c1) for c0, c1 in cs]

    spans = spans_of(mask, cols)
    if len(cols) == 2:
        # A 6x2 display shows 10 s: 5 s of limb leads, then 5 s of chest leads (D121). Its
        # grid is not a 5-mm paper grid (on the photos here a square is ~0.44 s), so the
        # time scale comes from the width of the traces in a column.
        col_px = float(np.mean([b - a for a, b in spans]))
        px_per_mm = col_px / (SCREEN_6X2_COLUMN_S * mm_per_s)
        scale_from = f"column = {SCREEN_6X2_COLUMN_S:g} s"
    else:
        pitch = _major_pitch_px(ratio, mask)
        if pitch is None or pitch < 8:
            raise ValueError("screen: grid pitch not found")
        px_per_mm = pitch / grid_mm
        scale_from = f"grid square = {grid_mm:g} mm"
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)

    def read(m: np.ndarray, spans: list[tuple[int, int]]) -> tuple[list[np.ndarray], list[float]]:
        rows, covs = [], []
        per_col = (len(lead_order) // len(cols)) if lead_order else None
        for a, b in spans:
            r, c = _read_rows(m, a, b, per_col, uv_per_px)
            rows += r
            covs += c
        return rows, covs

    rows, coverages = read(mask, spans)
    if _upright_fraction(rows) < SCREEN_UPRIGHT_MIN:          # upside down: turn by 180
        mask = mask[::-1, ::-1]
        w = mask.shape[1]
        cols = [(w - c1, w - c0) for c0, c1 in reversed(cols)]
        spans = spans_of(mask, cols)
        rows, coverages = read(mask, spans)
        turns = (turns + 2) % 4
    if lead_order is None:
        lead_order = (LIMB_6 + CHEST_6) if len(cols) == 2 and len(rows) == 12 else _guess_lead_order(rows)
    if len(lead_order) != len(rows):
        raise ValueError(f"screen: {len(rows)} rows for {len(lead_order)} leads")
    calib = {"layout": f"screen {len(rows) // len(cols)}x{len(cols)}", "mode": mode, "rotated_deg": 90 * turns,
             "px_per_mm": round(px_per_mm, 2), "mm_per_mv": mm_per_mv, "mm_per_s": mm_per_s,
             "scale_assumed": True, "scale_from": scale_from, "grid_mm_assumed": grid_mm,
             "panel_box": [int(v) for v in box], "leads": list(lead_order),
             "columns": [list(lead_order[i * (len(rows) // len(cols)):(i + 1) * (len(rows) // len(cols))])
                         for i in range(len(cols))]}
    digitize_screen.last_calibration = calib  # type: ignore[attr-defined]
    digitize_screen.last_coverage = float(np.mean(coverages))  # type: ignore[attr-defined]
    return dict(zip(lead_order, rows, strict=True))


def _guess_lead_order(rows: list[np.ndarray]) -> list[str]:
    if len(rows) == 12:
        return LIMB_6 + CHEST_6
    if len(rows) == 6:
        a, b, c = (r.astype(np.float64) for r in rows[:3])
        if np.std(b) > 0 and np.std(a + c) > 0 and np.corrcoef(b, a + c)[0, 1] >= SCREEN_LIMB_EINTHOVEN_R:
            return LIMB_6
        return CHEST_6
    return LIMB_6[:len(rows)]


def digitize_screen_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """12-lead SignalRepresentation; leads off the screen stay out of ``lead_windows``. A 6x2
    display is laid out like a page: the first column's leads valid over the first 5 s, the
    second column's over the next 5 s."""
    from cardiolens.contracts.signal import STANDARD_12_LEADS, Provenance

    traces = digitize_screen(path, **kw)
    calib = digitize_screen.last_calibration  # type: ignore[attr-defined]
    fs = float(calib["px_per_mm"] * calib["mm_per_s"])
    columns = calib["columns"]
    n_col = min(len(traces[ln]) for ln in calib["leads"])
    signal = np.zeros((12, n_col * len(columns)), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    for c, leads in enumerate(columns):
        for ln in leads:
            if ln in STANDARD_12_LEADS:
                signal[STANDARD_12_LEADS.index(ln), c * n_col:(c + 1) * n_col] = traces[ln][:n_col]
                windows[ln] = (c * n_col / fs, (c + 1) * n_col / fs)
    dur = signal.shape[1] / fs
    cov = digitize_screen.last_coverage  # type: ignore[attr-defined]
    note = (f"device screen photo: scale assumed ({calib['scale_from']}, {calib['mm_per_s']} mm/s, "
            f"{calib['mm_per_mv']} mm/mV); leads {', '.join(calib['leads'])} only; {dur:.1f} s")
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS), duration_s=float(dur),
        source="image",
        provenance=Provenance(digitized=True, confidence=round(cov, 3), lead_windows=windows,
                              notes=[f"{calib['layout']} screen: leads valid only inside their windows",
                                     f"calibration: {calib}", note]))


def render_device_screen(sig: SignalRepresentation, path: str | Path, *, leads: list[str] | None = None,
                         seconds: float = 2.5, px_per_mm: float = 9.0, mm_per_s: float = 25.0,
                         mm_per_mv: float = 10.0, row_pitch_mm: float = 15.0, dark: bool = False,
                         columns: int = 1) -> str:
    """A synthetic photo of a device screen (tests, benchmark): bezel, panel, grid, the
    leads as rows. ``dark``: bright green traces on a dark teal panel with a brighter grid.
    ``columns=2``: a 6x2 display — limb leads over [0, seconds), chest leads over
    [seconds, 2 x seconds), side by side; ``leads`` is then ignored."""
    import cv2

    groups = [LIMB_6, CHEST_6] if columns == 2 else [leads or CHEST_6]
    label = SCREEN_LABEL_FRACTION if columns == 1 else SCREEN_COLUMN_LABEL_FRACTION
    fs = sig.sampling_rate_hz
    col_px = int(seconds * mm_per_s * px_per_mm / (1 - label)) + SCREEN_RIGHT_MARGIN_PX
    width_px = col_px * len(groups)
    height_px = int((len(groups[0]) + 1) * row_pitch_mm * px_per_mm)
    bg, grid, ink, text = (((70, 60, 20), (90, 78, 30), (90, 255, 140), (230, 230, 230)) if dark else
                           ((235, 238, 232), (205, 208, 200), (25, 25, 25), (60, 60, 60)))
    panel = np.full((height_px, width_px, 3), bg, np.uint8)
    step = SCREEN_GRID_MM * px_per_mm
    for gx in np.arange(0, width_px, step):
        cv2.line(panel, (int(gx), 0), (int(gx), height_px), grid, 1)
    for gy in np.arange(0, height_px, step):
        cv2.line(panel, (0, int(gy)), (width_px, int(gy)), grid, 1)
    n = int(seconds * fs)
    for c, group in enumerate(groups):
        x_left = c * col_px + int(label * col_px)
        xs = x_left + np.arange(n) / fs * mm_per_s * px_per_mm
        for i, ln in enumerate(group):
            y_base = (i + 1) * row_pitch_mm * px_per_mm
            cv2.putText(panel, ln, (c * col_px + 4, int(y_base) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text, 1)
            v = sig.lead(ln)[c * n:(c + 1) * n].astype(np.float64)
            ys = y_base - (v - np.median(v)) / 1000.0 * mm_per_mv * px_per_mm
            cv2.polylines(panel, [np.stack([xs, ys], axis=1).astype(np.int32)], False, ink, 2)
    img = np.full((height_px + 260, width_px + 120, 3), (40, 32, 30), np.uint8)   # bezel and room
    img[60:60 + height_px, 60:60 + width_px] = panel
    if dark:   # the recorders' blue button bar along the bottom of the display (D122)
        cv2.rectangle(img, (60, 60 + height_px + 10), (60 + width_px, 60 + height_px + 90), (200, 110, 30), -1)
    else:
        cv2.rectangle(img, (60, 60 + height_px + 20), (60 + width_px // 2, 60 + height_px + 150), (235, 235, 235), -1)
    cv2.imwrite(str(path), img)
    return str(path)
