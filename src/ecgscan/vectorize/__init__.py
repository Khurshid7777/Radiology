"""Read every panel of a framed page into ScanResult traces (TZ §5.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.contract import Trace
from ecgscan.layout.panels import Panel
from ecgscan.segmentation import TraceMask, thin
from ecgscan.validate import lead_quality
from ecgscan.vectorize.trace import estimate_stroke, refine_steep, resample_nan, trace_panel


def legacy_track(ink: np.ndarray, skel: np.ndarray, p: Panel, pitch_px: float) -> np.ndarray:
    """The CardioLens D47/D52 tracker on this panel (box = one row pitch around the
    baseline; it extends it by half itself). Second vectorisation method (TZ §5.7)."""
    from cardiolens.layer1_ingestion.digitize import _trace_rows_tracked

    h, w = ink.shape
    x0, x1 = max(0, p.x0), min(w, p.x1)
    if x1 <= x0:
        return np.full(max(0, p.x1 - p.x0), np.nan)
    y0, y1 = int(round(p.baseline - pitch_px / 2)), int(round(p.baseline + pitch_px / 2))
    out = _trace_rows_tracked(ink, skel, x0, x1, max(0, y0), min(h, y1))
    pad_l = x0 - p.x0
    return np.concatenate([np.full(pad_l, np.nan), out]) if pad_l > 0 else out


def fuse(path: np.ndarray, detail: np.ndarray, pitch_px: float, *, tol_pitches: float = 0.15,
         margin: int = 10, max_short: int = 8, far_pitches: float = 0.35, max_gap_pitches: float = 1.0,
         apex_px: float | None = None, baseline: float | None = None, other_baselines: list[float] | None = None) -> np.ndarray:
    """Two methods, one trace (TZ §5.7: choose by quality, per stretch).

    ``path`` (the DP tracer) holds the right row through crossings and noise; ``detail``
    (the D47 tracker: skeleton edges, apex from the raw stroke) gets steep QRS edges and
    peaks right but on photos can jump to a neighbouring row. ``detail`` is kept unless it
    *left the row*: a disagreement longer than ``max_short`` columns (a QRS apex where
    the path is short of the tip lasts a few columns), or a stretch where ``detail`` sits
    nearer another row's baseline than this one. Around such a stretch (± ``margin``)
    the path is used. Where the two differ by more than ``apex_px`` (a small R at 5 mm/mV
    is within the row tolerance) the value further from the isoline wins.
    """
    n = min(len(path), len(detail))
    out = path.copy()
    if n == 0:
        return out
    a, b = path[:n], detail[:n]
    fin = np.isfinite(a) & np.isfinite(b)
    differ = fin & (np.abs(a - b) > tol_pitches * pitch_px)
    left = np.zeros(n, dtype=bool)
    if differ.any():
        idx = np.flatnonzero(differ)
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(seg) > max_short:
                left[seg] = True
    # Far from the path at any length: with apexes read from the raw stroke the path
    # reaches the tips itself, so a large gap means header text, a label or another row.
    left |= fin & (np.abs(a - b) > far_pitches * pitch_px)
    if baseline is not None and other_baselines:
        own_d = np.abs(b - baseline)
        oth_d = np.min(np.abs(b[:, None] - np.asarray(other_baselines)[None, :]), axis=1)
        left |= np.isfinite(b) & (oth_d < own_d) & (oth_d < 0.2 * pitch_px)
    # Where the path has a gap, ``detail`` has nothing to be checked against. It is kept
    # only inside a short gap bounded on both sides by columns where the two agree (a
    # steep QRS edge the path skipped); a gap open at a panel end, a long one, or one
    # next to a stretch where ``detail`` left the row is header text or a label.
    gap = ~np.isfinite(a) & np.isfinite(b)
    if gap.any():
        agree = fin & ~left
        idx = np.flatnonzero(~np.isfinite(a))
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            lo, hi = seg[0] - 1, seg[-1] + 1
            bounded = lo >= 0 and hi < n and agree[lo] and agree[hi]
            if not bounded or len(seg) > max_gap_pitches * pitch_px:
                left[seg] = True
    if left.any() and margin > 0:
        left = np.convolve(left.astype(float), np.ones(2 * margin + 1), mode="same") > 0
    use = np.isfinite(b) & ~left
    # A short disagreement is an apex one method stopped short of: the value further from
    # the isoline wins (the path itself reaches a spike the D47 tracker cut through).
    if np.isfinite(a).any():
        iso = float(np.nanmedian(a))
        apart = fin & (np.abs(a - b) > (apex_px if apex_px is not None else tol_pitches * pitch_px))
        use &= ~(apart & (np.abs(a - iso) > np.abs(b - iso)))
    out[:n][use] = b[use]
    return out


@dataclass
class PanelReading:
    panel: Panel
    trace: Trace
    y_px: np.ndarray            # chosen y per column (rectified image), for the overlay
    quality: float
    ambiguity: float
    edge_contact: float


def read_panels(mask: TraceMask, panels: list[Panel], *, px_per_mm: float, speed_mm_s: float, gain_mm_mv: float,
                pitch_px: float, fs_out: float, cfg: dict) -> list[PanelReading]:
    stroke = estimate_stroke(mask.ink)
    skel = thin(mask.ink) if cfg.get("skeleton", True) else None
    band = cfg.get("band_pitches", 1.3) * pitch_px
    fs_col = px_per_mm * speed_mm_s
    out = []
    rows = sorted({p.baseline for p in panels})
    method = cfg.get("method", "dp_path")
    for p in panels:
        pt = trace_panel(mask.ink, mask.darkness, p.x0, p.x1, int(p.baseline - band), int(p.baseline + band),
                         p.baseline, stroke_px=stroke, pitch_px=pitch_px,
                         baseline_weight=cfg.get("baseline_weight", 0.6), gap_cost=cfg.get("gap_cost", 2.5),
                         skeleton=skel, other_baselines=[b for b in rows if b != p.baseline],
                         neighbour_cost=cfg.get("neighbour_cost", 4.0))
        y = pt.y
        if method == "legacy_track":
            y = legacy_track(mask.ink, skel, p, pitch_px)
        elif method == "hybrid":
            y = fuse(y, legacy_track(mask.ink, skel, p, pitch_px), pitch_px,
                     tol_pitches=cfg.get("fuse_tol_pitches", 0.15), margin=cfg.get("fuse_margin_cols", 10),
                     max_short=cfg.get("fuse_max_short_cols", 8),
                     max_gap_pitches=cfg.get("fuse_max_gap_pitches", 0.5),
                     apex_px=cfg.get("fuse_apex_strokes", 2.0) * stroke, baseline=p.baseline,
                     other_baselines=[b for b in rows if b != p.baseline])
        if cfg.get("steep_rows", False):
            y = refine_steep(y, mask.ink, p.x0, stroke, min_slope_strokes=cfg.get("steep_min_slope_strokes", 1.0),
                             max_run_strokes=cfg.get("steep_max_run_strokes", 4.0),
                             tip_cols=cfg.get("steep_tip_cols", 3))
        n_expected = p.x1 - p.x0
        if len(y) < n_expected:          # the panel runs past the image: the rest is missing, not zero
            y = np.concatenate([y, np.full(n_expected - len(y), np.nan)])
        iso = float(np.nanmedian(y)) if np.isfinite(y).any() else p.baseline
        mv = (iso - y) / (px_per_mm * gain_mm_mv)
        samples = resample_nan(mv, fs_col, fs_out, p.t1_s - p.t0_s)
        miss = float(np.mean(~np.isfinite(samples))) if samples.size else 1.0
        q = lead_quality(miss, pt.ambiguity, pt.edge_contact)
        flags = []
        if miss > cfg.get("gap_flag", 0.05):
            flags.append("TRACE_GAPS")
        if pt.edge_contact > cfg.get("overlap_flag", 0.1):
            flags.append("LEAD_OVERLAP")
        tr = Trace.from_array(samples, start_s=p.t0_s, fs=fs_out, lead=p.lead, quality=round(q, 3),
                              coverage=round(1.0 - miss, 4), method=method,
                              box_px=[int(p.x0), int(p.baseline - band), int(p.x1), int(p.baseline + band)], flags=flags)
        out.append(PanelReading(p, tr, y, q, pt.ambiguity, pt.edge_contact))
    return out
