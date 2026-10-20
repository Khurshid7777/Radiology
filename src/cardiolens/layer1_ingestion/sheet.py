"""Report sheets: twelve rows in a block, text beside it, a rhythm strip below (D126).

Cardipia 800H and similar recorders print an A4/A5 report: the twelve leads as
rows of ~4.7 s at 25 mm/s in the left half, patient data and the device's own
measurements on the right, and one rhythm lead over 10 s at 12.5 mm/s under the
text. A phone photo of it is usually turned by a quarter.

1. the turn: of the candidate turns (both quarters for a portrait photo, 0 / 180
   degrees otherwise), the one where the row block starts at the left and aVR
   points down;
2. the block: ``digitize_strip`` in block mode — the rows' extent is where most of
   them carry ink (``block_fraction``), a title line above the first row does not
   take its place, the baselines are lines fitted over the block;
3. the rhythm strip: the row of most ink to the right of the block at the height of
   the last rows. At 12.5 mm/s a QRS is one or two pixels wide and the tracker cuts
   it, so the strip is read as the ink farthest from its baseline per column
   (``rhythm_envelope``);
4. the record: the block's leads valid over its own window, lead II continued by
   the rhythm strip to 10 s — close to a 3x4 page with its lead-II strip. The strip
   is taken to start with the block (the recorder prints one acquisition); the join
   is marked as a discontinuity, so no RR interval spans it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import STANDARD_12_LEADS, Provenance, SignalRepresentation

SHEET_BLOCK_FRACTION = 0.5     # of all rows carrying ink near their baselines: the block
SHEET_INK_RATIOS = (0.78, 0.7)
SHEET_MAX_BLOCK_WIDTH = 0.75   # of the sheet: a wider block is a plain 12x1 strip
SHEET_MAX_BLOCK_START = 0.3    # of the sheet: the block starts in the left part
SHEET_RHYTHM_MM_PER_S = 12.5
SHEET_RHYTHM_SEARCH_PITCHES = 1.2   # the strip's baseline lies within this of the last row's
SHEET_RHYTHM_UP_MM = 15.0      # the envelope looks this far above the baseline ...
SHEET_RHYTHM_DOWN_MM = 5.0     # ... and this far below (the time-axis text sits lower)
SHEET_RHYTHM_MIN_S = 4.0       # a shorter run of ink is not a rhythm strip
SHEET_BRIDGE_MM = 3.0          # a dropout in the strip's ink up to this long is bridged
SHEET_GAP_MM = 5.0             # the strip starts this far right of the block at least


def rhythm_envelope(mask: np.ndarray, y_base: int, px_per_mm: float) -> np.ndarray:
    """Signed distance (px, up positive) of the ink farthest from ``y_base`` per column of a
    band around it; columns without ink read 0."""
    lo = max(0, y_base - int(SHEET_RHYTHM_UP_MM * px_per_mm))
    hi = min(mask.shape[0], y_base + int(SHEET_RHYTHM_DOWN_MM * px_per_mm))
    sub = mask[lo:hi]
    ys = np.arange(lo, hi)[:, None]
    has = sub.any(axis=0)
    top = np.where(has, np.where(sub, ys, 10 ** 6).min(axis=0), y_base)
    bot = np.where(has, np.where(sub, ys, -1).max(axis=0), y_base)
    up, down = y_base - top, bot - y_base
    return np.where(up >= down, up, -down).astype(np.float64)


def _read_rhythm(geom: dict, px_per_mm: float, ink_ratio: float) -> tuple[np.ndarray, dict] | None:
    """The rhythm strip right of the block as an envelope in px at ``px_per_mm`` x 12.5 Hz."""
    from cardiolens.layer1_ingestion.strip import _keep_long_components, ink_mask_colour

    bgr, base, pitch, box = geom["bgr"], geom["base"], geom["pitch"], geom["box"]
    x0 = geom["trace_x"][1] + int(SHEET_GAP_MM * px_per_mm)
    x1 = box[2]
    if x1 - x0 < SHEET_RHYTHM_MIN_S * SHEET_RHYTHM_MM_PER_S * px_per_mm:
        return None
    mask = ink_mask_colour(bgr, px_per_mm, ink_ratio)
    mask = _keep_long_components(mask, int(2 * px_per_mm))[:, x0:x1]
    y_last = float(np.median(base[-1, x0:x1]))
    lo = max(0, int(y_last - SHEET_RHYTHM_SEARCH_PITCHES * pitch))
    hi = min(mask.shape[0], int(y_last + SHEET_RHYTHM_SEARCH_PITCHES * pitch))
    prof = mask[lo:hi].sum(axis=1)
    if prof.max() == 0:
        return None
    y_base = lo + int(np.argmax(prof))
    env = rhythm_envelope(mask, y_base, px_per_mm)
    near = mask[max(0, y_base - int(pitch / 3)):y_base + int(pitch / 3)].any(axis=0)
    cols = np.flatnonzero(near)
    if cols.size < 2:
        return None
    # The strip's extent: the longest run of columns with ink near the baseline,
    # bridging only small dropouts (a pause is flat but drawn; the end of the last
    # row of the block, at the same height, lies beyond a wider gap).
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > SHEET_BRIDGE_MM * px_per_mm) + 1)
    run = max(runs, key=len)
    a, b = int(run[0]), int(run[-1]) + 1
    if b - a < SHEET_RHYTHM_MIN_S * SHEET_RHYTHM_MM_PER_S * px_per_mm:
        return None
    return env[a:b], {"rhythm_x": (x0 + a, x0 + b), "rhythm_y": y_base}


def _block_polarity(traces: dict[str, np.ndarray]) -> float:
    """+1 per expected QRS direction: II up, aVR down."""
    def pol(x: np.ndarray) -> float:
        med = float(np.median(x))
        return float(np.sign((np.percentile(x, 99) - med) - (med - np.percentile(x, 1))))
    return pol(traces["II"]) - pol(traces["aVR"])


def digitize_sheet(path: str | Path, *, mm_per_s: float = 25.0) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """``({lead: block trace uV}, rhythm II uV or None)``; ``digitize_sheet.last_calibration``
    holds the provenance (``fs`` of the block, ``rhythm_fs``, the turn)."""
    import cv2

    from cardiolens.layer1_ingestion.strip import digitize_strip

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    turns = (1, 3) if bgr.shape[0] > bgr.shape[1] else (0, 2)
    best = None
    for t in turns:
        view = np.ascontiguousarray(np.rot90(bgr, -t)) if t else bgr
        for ratio in SHEET_INK_RATIOS:
            try:
                traces = digitize_strip(view, rows=12, block_fraction=SHEET_BLOCK_FRACTION, ink_ratio=ratio,
                                        mm_per_s=mm_per_s)
            except ValueError:
                continue
            calib = dict(digitize_strip.last_calibration)  # type: ignore[attr-defined]
            geom = digitize_strip.last_geometry  # type: ignore[attr-defined]
            x_left, x_right = calib["trace_x"]
            width = geom["box"][2] - geom["box"][0]
            if (x_right - x_left) > SHEET_MAX_BLOCK_WIDTH * width or x_left > SHEET_MAX_BLOCK_START * width:
                continue
            ppm = calib["px_per_mm"]
            rhythm = _read_rhythm(geom, ppm, ratio)
            if rhythm is None:      # a narrow block alone is not enough to call it a sheet
                continue
            score = _block_polarity(traces)
            if best is None or score > best[0]:
                best = (score, t, ratio, traces, calib, rhythm,
                        float(digitize_strip.last_coverage))  # type: ignore[attr-defined]
    if best is None:
        raise ValueError("sheet: no twelve-row block with a rhythm strip beside it found")
    score, t, ratio, traces, calib, rhythm, cov = best
    ppm = calib["px_per_mm"]
    uv_per_px = 1000.0 / (ppm * calib["mm_per_mv"])
    rhythm_uv = None
    if rhythm is not None:
        env, where = rhythm
        rhythm_uv = (env - float(np.median(env))) * uv_per_px
        calib.update(where)
    calib.update(layout="12x1 sheet", rotated_deg=90 * t, ink_ratio=ratio, match_score=round(score, 2),
                 fs=float(ppm * mm_per_s), rhythm_fs=float(ppm * SHEET_RHYTHM_MM_PER_S),
                 rhythm_mm_per_s=SHEET_RHYTHM_MM_PER_S, mm_per_s=mm_per_s)
    digitize_sheet.last_calibration = calib  # type: ignore[attr-defined]
    digitize_sheet.last_coverage = cov  # type: ignore[attr-defined]
    return traces, rhythm_uv


def digitize_sheet_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """12-lead SignalRepresentation: the block's leads over the block's window; lead II
    continued by the rhythm strip to its 10 s."""
    traces, rhythm = digitize_sheet(path, **kw)
    calib = digitize_sheet.last_calibration  # type: ignore[attr-defined]
    fs = calib["fs"]
    n_block = min(len(v) for v in traces.values())
    n_rhythm = 0
    if rhythm is not None:
        t = np.arange(int(len(rhythm) / calib["rhythm_fs"] * fs)) / fs
        rhythm = np.interp(t, np.arange(len(rhythm)) / calib["rhythm_fs"], rhythm).astype(np.float32)
        n_rhythm = len(rhythm)
    n = max(n_block, n_rhythm)
    signal = np.zeros((12, n), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    for ln, tr in traces.items():
        signal[STANDARD_12_LEADS.index(ln), :n_block] = tr[:n_block]
        windows[ln] = (0.0, n_block / fs)
    cuts: list[float] = []
    if rhythm is not None and n_rhythm > n_block:
        # Lead II: the block's own row over the block (read like the other rows, so the
        # page checks compare like with like), the rhythm strip after it. The join is a
        # discontinuity: the strip is only taken to start with the rows.
        ii = STANDARD_12_LEADS.index("II")
        tail = rhythm[n_block:] - float(np.median(rhythm[n_block:])) + float(np.median(traces["II"][:n_block]))
        signal[ii, n_block:n_rhythm] = tail
        windows["II"] = (0.0, n_rhythm / fs)
        cuts.append(n_block / fs)
    note = (f"report sheet photo: 12 rows over {n_block / fs:.1f} s at {calib['mm_per_s']} mm/s"
            + (f"; lead II continued from the rhythm strip to {n_rhythm / fs:.1f} s "
               f"({calib['rhythm_mm_per_s']} mm/s assumed, taken to start with the rows)" if cuts else ""))
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS), duration_s=n / fs, source="image",
        provenance=Provenance(digitized=True, confidence=round(digitize_sheet.last_coverage, 3),  # type: ignore[attr-defined]
                              lead_windows=windows, discontinuities_s=cuts or None,
                              notes=[f"calibration: {calib}", note]))


def render_report_sheet(sig: SignalRepresentation, path: str | Path, *, px_per_mm: int = 5,
                        block_s: float = 4.7, mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                        row_pitch_mm: float = 15.0, turn: int = 0) -> str:
    """A synthetic report sheet (tests): a title line, twelve rows of ``block_s`` at the left,
    lines of text on the right, the rhythm lead II over 10 s at 12.5 mm/s under the text,
    a time axis; ``turn`` quarter turns clockwise."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import _draw_grid, _draw_trace
    from cardiolens.layer1_ingestion.strip import STRIP_12_ROWS

    fs = sig.sampling_rate_hz
    left = int(12 * px_per_mm)
    block_w = int(block_s * mm_per_s * px_per_mm)
    rhythm_w = int(10.0 * SHEET_RHYTHM_MM_PER_S * px_per_mm)
    gap = int(8 * px_per_mm)
    w = left + block_w + gap + rhythm_w + int(10 * px_per_mm)
    pitch = int(row_pitch_mm * px_per_mm)
    top = int(14 * px_per_mm)
    h = top + pitch * 12 + int(16 * px_per_mm)
    img = np.full((h, w, 3), 255, np.uint8)
    _draw_grid(img, px_per_mm)
    ink, scale = (30, 30, 30), 0.09 * px_per_mm
    cv2.putText(img, "ECG Rest [Auto]   09:13  Oct 18 2026   Hospital   RECID 541", (left, int(6 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    n = int(block_s * fs)
    for r, ln in enumerate(STRIP_12_ROWS):
        y = top + pitch * r + pitch // 2
        cv2.putText(img, ln, (int(2 * px_per_mm), y - int(3 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1)
        _draw_trace(img, sig.lead(ln)[:n], fs, left, y, mm_per_s * px_per_mm, px_per_uv)
    x_text = left + block_w + gap
    for i, line in enumerate(["ID: 20261018.2112", "Name:", "Date of birth:", "Sex:",
                              "HR: 96bpm PR: 132ms QRS: 110ms QT: 378ms", "RV5: 1.76mV SV1: --mV",
                              "[07] Arrhythmia: variable RR interval", "Checked: Doctor"]):
        cv2.putText(img, line, (x_text, top + int((6 + 9 * i) * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1)
    y_r = top + pitch * 11 + pitch // 2
    _draw_trace(img, sig.lead("II")[:int(10.0 * fs)], fs, x_text, y_r, SHEET_RHYTHM_MM_PER_S * px_per_mm, px_per_uv)
    for s in range(11):
        cv2.putText(img, f"{s}s", (x_text + int(s * SHEET_RHYTHM_MM_PER_S * px_per_mm), y_r + int(12 * px_per_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1)
    for _ in range(turn % 4):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(path), img)
    return str(path)
