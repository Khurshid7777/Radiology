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


def is_device_screen(bgr: np.ndarray) -> bool:
    """A lit panel smaller than the photo with no red paper grid in it."""
    box = find_screen_panel(bgr)
    if box is None:
        return False
    x0, y0, x1, y1 = box
    p = bgr[y0:y1, x0:x1].astype(int)
    red = (p[..., 2] - p[..., 0] > 40) & (p[..., 2] > 150)
    return bool(red.mean() <= SCREEN_MAX_RED_FRACTION)


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
    top = max(f[i] for i in loc)
    k = min(i for i in loc if f[i] >= SCREEN_FUNDAMENTAL_FRACTION * top)   # the fundamental, not a harmonic
    return float(1.0 / fr[k])


def digitize_screen(path: str | Path, *, lead_order: list[str] | None = None,
                    mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                    grid_mm: float = SCREEN_GRID_MM) -> dict[str, np.ndarray]:
    """``{lead: trace_uv}`` for a photo of a device screen; ``digitize_screen.last_calibration``
    and ``last_coverage`` hold the provenance. Raises ValueError when no panel or rows are found."""
    import cv2
    from scipy.signal import find_peaks

    from cardiolens.layer1_ingestion.digitize import _fill_gaps, _trace_rows_tracked, thin_mask
    from cardiolens.layer1_ingestion.strip import straighten

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    box = find_screen_panel(bgr)
    if box is None:
        raise ValueError("screen: no lit panel found")
    x0, y0, x1, y1 = box
    panel = bgr[y0:y1, x0:x1]
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, SCREEN_BG_KERNEL).astype(np.float32) + 1.0
    ratio = g.astype(np.float32) / bg
    mask = ratio < SCREEN_INK_RATIO
    pitch = _major_pitch_px(ratio, mask)
    if pitch is None or pitch < 8:
        raise ValueError("screen: grid pitch not found")
    px_per_mm = pitch / grid_mm
    w = mask.shape[1]
    x_left, x_right = int(SCREEN_LABEL_FRACTION * w), w - SCREEN_RIGHT_MARGIN_PX
    prof = mask[:, x_left:x_right].sum(axis=1).astype(float)
    n_rows = len(lead_order) if lead_order else None
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
    t0, t1 = half - int(0.75 * row_pitch), half + int(0.75 * row_pitch)
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    rows, coverages = [], []
    for b in base:
        band = straighten(mask, np.full(w, float(b)), half)
        raw = _trace_rows_tracked(band, thin_mask(band), x_left, x_right, t0, t1, baseline_weight=1.0)
        trace, cov = _fill_gaps(raw)
        rows.append(((float(np.median(trace)) - trace) * uv_per_px).astype(np.float32))
        coverages.append(cov)
    if lead_order is None:
        lead_order = _guess_lead_order(rows)
    calib = {"layout": f"screen {n_rows}x1", "px_per_mm": round(px_per_mm, 2), "mm_per_mv": mm_per_mv,
             "mm_per_s": mm_per_s, "scale_assumed": True, "grid_mm_assumed": grid_mm,
             "panel_box": [int(v) for v in box], "leads": list(lead_order)}
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
    """12-lead SignalRepresentation; leads off the screen stay out of ``lead_windows``."""
    from cardiolens.layer1_ingestion.strip import strip_to_signal

    traces = digitize_screen(path, **kw)
    calib = digitize_screen.last_calibration  # type: ignore[attr-defined]
    sig = strip_to_signal(traces, px_per_mm=calib["px_per_mm"], mm_per_s=calib["mm_per_s"],
                          coverage=digitize_screen.last_coverage,  # type: ignore[attr-defined]
                          calibration=calib)
    sig.provenance.notes.append(
        f"device screen photo: scale assumed ({calib['grid_mm_assumed']} mm grid, {calib['mm_per_s']} mm/s, "
        f"{calib['mm_per_mv']} mm/mV); leads {', '.join(calib['leads'])} only; {sig.duration_s:.1f} s")
    return sig


def render_device_screen(sig: SignalRepresentation, path: str | Path, *, leads: list[str] | None = None,
                         seconds: float = 2.5, px_per_mm: float = 9.0, mm_per_s: float = 25.0,
                         mm_per_mv: float = 10.0, row_pitch_mm: float = 15.0) -> str:
    """A synthetic photo of a device screen: dark bezel, lit panel, grey 5-mm grid,
    the leads as rows from ``SCREEN_LABEL_FRACTION`` of the width (tests, benchmark)."""
    import cv2

    leads = leads or CHEST_6
    fs = sig.sampling_rate_hz
    width_px = int(seconds * mm_per_s * px_per_mm / (1 - SCREEN_LABEL_FRACTION)) + SCREEN_RIGHT_MARGIN_PX
    height_px = int((len(leads) + 1) * row_pitch_mm * px_per_mm)
    panel = np.full((height_px, width_px, 3), (235, 238, 232), np.uint8)
    step = SCREEN_GRID_MM * px_per_mm
    for gx in np.arange(0, width_px, step):
        cv2.line(panel, (int(gx), 0), (int(gx), height_px), (205, 208, 200), 1)
    for gy in np.arange(0, height_px, step):
        cv2.line(panel, (0, int(gy)), (width_px, int(gy)), (205, 208, 200), 1)
    x_left = int(SCREEN_LABEL_FRACTION * width_px)
    n = int(seconds * fs)
    xs = x_left + np.arange(n) / fs * mm_per_s * px_per_mm
    for i, ln in enumerate(leads):
        y_base = (i + 1) * row_pitch_mm * px_per_mm
        cv2.putText(panel, ln, (6, int(y_base) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
        v = sig.lead(ln)[:n].astype(np.float64)
        ys = y_base - (v - np.median(v)) / 1000.0 * mm_per_mv * px_per_mm
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(panel, [pts], False, (25, 25, 25), 2)
    img = np.full((height_px + 260, width_px + 120, 3), (40, 32, 30), np.uint8)   # bezel and room
    img[60:60 + height_px, 60:60 + width_px] = panel
    cv2.rectangle(img, (60, 60 + height_px + 20), (60 + width_px // 2, 60 + height_px + 150), (235, 235, 235), -1)
    cv2.imwrite(str(path), img)
    return str(path)
