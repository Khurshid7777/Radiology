"""Read every panel of a framed page into ScanResult traces (TZ §5.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.contract import Trace
from ecgscan.layout.panels import Panel
from ecgscan.segmentation import TraceMask, thin
from ecgscan.validate import lead_quality
from ecgscan.vectorize.trace import estimate_stroke, resample_nan, trace_panel


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
    for p in panels:
        pt = trace_panel(mask.ink, mask.darkness, p.x0, p.x1, int(p.baseline - band), int(p.baseline + band),
                         p.baseline, stroke_px=stroke, pitch_px=pitch_px,
                         baseline_weight=cfg.get("baseline_weight", 0.6), gap_cost=cfg.get("gap_cost", 2.5),
                         skeleton=skel, other_baselines=[b for b in rows if b != p.baseline],
                         neighbour_cost=cfg.get("neighbour_cost", 4.0))
        y = pt.y
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
