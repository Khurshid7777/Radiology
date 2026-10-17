"""Read every panel of a framed page into ScanResult traces (TZ §5.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.contract import Trace
from ecgscan.layout.panels import Panel
from ecgscan.segmentation import TraceMask, thin
from ecgscan.validate import lead_quality
from ecgscan.vectorize.trace import estimate_stroke, resample_nan, trace_panel


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
         margin: int = 10) -> np.ndarray:
    """Two methods, one trace (TZ §5.7: choose by quality, per stretch).

    ``path`` (the DP tracer) holds the right row through crossings and noise; ``detail``
    (the D47 tracker: skeleton edges, apex from the raw stroke) gets steep QRS edges and
    peaks right but on photos can jump to a neighbouring row. ``detail`` is used wherever
    it stays within ``tol_pitches`` of the path, ``path`` elsewhere (± ``margin``
    columns around any disagreement, so a half-jump is not kept).
    """
    n = min(len(path), len(detail))
    out = path.copy()
    if n == 0:
        return out
    a, b = path[:n], detail[:n]
    agree = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= tol_pitches * pitch_px)
    bad = np.isfinite(b) & ~agree
    if bad.any() and margin > 0:
        k = np.ones(2 * margin + 1)
        bad = np.convolve(bad.astype(float), k, mode="same") > 0
    use = agree & ~bad
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
                     tol_pitches=cfg.get("fuse_tol_pitches", 0.15), margin=cfg.get("fuse_margin_cols", 10))
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
                              coverage=round(1.0 - miss, 4), method="dp_path",
                              box_px=[int(p.x0), int(p.baseline - band), int(p.x1), int(p.baseline + band)], flags=flags)
        out.append(PanelReading(p, tr, y, q, pt.ambiguity, pt.edge_contact))
    return out
