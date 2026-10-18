"""ecgscan pipeline: image -> ScanResult (TZ §5).

Intake, grid and resolution check, rectification to a known px/mm, calibration
from pulse + header OCR (M2); segmentation, registry panels, vectorisation and
physiological validation (M3).

Status rules:
* ``rejected`` — LOW_RESOLUTION, GRID_NOT_FOUND, LOW_CONTRAST, LAYOUT_UNKNOWN,
  TRACE_BROKEN, PROCESSING_ERROR (no traces are returned);
* ``needs_confirmation`` — speed or gain not established (SD-011), sources disagree,
  layout uncertain, limb leads inconsistent, a lead largely missing, the CardioLens
  beats disagreeing with the page columns (D92), or overall quality below
  ``validate.ok_min_overall``;
* ``ok`` — otherwise.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecgscan import PIPELINE_VERSION
from ecgscan.calibration.header_ocr import read_bands
from ecgscan.calibration.pulse import find_pulses, summarize_pulses
from ecgscan.calibration.resolve import resolve
from ecgscan.config import load_yaml
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source
from ecgscan.geometry.grid import estimate_grid
from ecgscan.geometry.rectify import Rectified, rectify
from ecgscan.io import load_image
from ecgscan.layout import get_layout
from ecgscan.layout.detect import detect_layout
from ecgscan.layout.panels import frame_page, panels
from ecgscan.layout.rows import long_stroke_mask
from ecgscan.segmentation import make_segmenter
from ecgscan.validate import beats_agree, limb_consistency
from ecgscan.vectorize import read_panels


@dataclass
class ScanHints:
    """What the client or the doctor already knows (TZ §6.3: confirmation and corrections)."""

    paper_speed_mm_s: float | None = None
    gain_mm_mv: float | None = None
    layout: str | None = None
    client: str | None = None
    device: str | None = None
    header_masked: bool = False


@dataclass
class ScanDebug:
    """Intermediate results for artifacts and the bench (TZ §5.9)."""

    rectified: Rectified | None = None
    pulses: list = field(default_factory=list)
    ocr_texts: list[str] = field(default_factory=list)
    rows: int | None = None
    frame: object | None = None
    readings: list = field(default_factory=list)
    consistency: object | None = None
    timings: dict[str, float] = field(default_factory=dict)


def pipeline_config() -> dict:
    return load_yaml("pipeline.yaml")


def trace_contrast(bgr: np.ndarray, px_per_mm: float) -> float:
    """1 - (darkest neutral ink / paper): ~0.9 for a fresh print, low for faded thermal paper."""
    import cv2

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    neutral = g[chroma < 40]
    if neutral.size < 100:
        return 0.0
    paper = float(np.percentile(g, 90))
    ink = float(np.percentile(neutral, 0.2))
    return max(0.0, 1.0 - ink / max(paper, 1.0))


def scan(image: str | Path | bytes, *, hints: ScanHints | None = None, fs_out: float = 500.0,
         debug: ScanDebug | None = None, artifacts_dir: str | Path | None = None,
         config: dict | None = None) -> ScanResult:
    """Digitise one photo/scan. ``artifacts_dir``: write rectified / mask / overlay images there."""
    t_start = time.perf_counter()
    cfg = config or pipeline_config()
    hints = hints or ScanHints()
    dbg = debug if debug is not None else ScanDebug()
    q_cfg, g_cfg = cfg["quality"], cfg["geometry"]

    def tick(name: str, t0: float) -> float:
        dbg.timings[name] = round(time.perf_counter() - t0, 3)
        return time.perf_counter()

    img = load_image(image)
    source = Source(modality="photo", original_sha256=img.original_sha256, original_bytes=img.original_bytes,
                    mime_type=img.mime_type, width_px=img.width_px, height_px=img.height_px,
                    exif_orientation=img.exif_orientation, client=hints.client, device=hints.device,
                    header_masked=hints.header_masked, pipeline_version=PIPELINE_VERSION)
    reasons: list[str] = []
    warnings: list[str] = []
    # The phone's frame checks, repeated on the photo as received (TZ §6.1, SD-046):
    # recorded for the record and for comparing client and server, not a gate.
    frame_qc = None
    if q_cfg.get("capture_qc", True):
        from ecgscan.capture_qc import check_frame

        frame_qc = check_frame(img.bgr[..., ::-1], grid=False)

    def result(status: str, *, calibration: Calibration | None = None, layout: Layout | None = None,
               leads=None, rhythm=None, overall: float = 0.0, ppm: float | None = None,
               contrast: float | None = None) -> ScanResult:
        return ScanResult(
            scan_id=str(uuid.uuid4()), status=status, reject_reasons=sorted(set(reasons)) if status == "rejected" else [],
            warnings=sorted(set(warnings)), source=source,
            calibration=calibration or Calibration(confidence=0.0), layout=layout or Layout(confidence=0.0),
            sampling_rate_hz=fs_out, leads=leads or {}, rhythm=rhythm or [],
            quality=QualitySummary(overall=round(float(overall), 3), px_per_mm=None if ppm is None else round(ppm, 3),
                                   trace_contrast=None if contrast is None else round(contrast, 3),
                                   sharpness=None if frame_qc is None else frame_qc.sharpness,
                                   glare_fraction=None if frame_qc is None else frame_qc.glare,
                                   processing_ms=int((time.perf_counter() - t_start) * 1000)))

    # 1. Grid and resolution on the photo as taken (TZ §5.3).
    t = time.perf_counter()
    raw = estimate_grid(img.bgr)
    t = tick("grid", t)
    raw_ppm = raw.px_per_mm
    if raw_ppm is None or raw.confidence < q_cfg["min_grid_confidence"]:
        reasons.append("GRID_NOT_FOUND")
        return result("rejected", ppm=raw_ppm)
    if raw_ppm < q_cfg["min_px_per_mm"]:
        reasons.append("LOW_RESOLUTION")
        return result("rejected", ppm=raw_ppm)
    if raw_ppm < q_cfg["marginal_px_per_mm"]:
        warnings.append("LOW_RESOLUTION_MARGINAL")

    # 2. Rectify to the working scale.
    try:
        rect = rectify(img.bgr, working_px_per_mm=float(g_cfg["working_px_per_mm"]), raw_grid=raw,
                       max_upscale=float(g_cfg["max_upscale"]), fit_px_per_mm=g_cfg.get("fit_px_per_mm"))
    except Exception:
        reasons.append("GRID_NOT_FOUND")
        return result("rejected", ppm=raw_ppm)
    dbg.rectified = rect
    t = tick("rectify", t)
    ppm = rect.px_per_mm
    contrast = trace_contrast(rect.bgr, ppm)
    if contrast < q_cfg.get("min_trace_contrast", 0.0):
        reasons.append("LOW_CONTRAST")
        return result("rejected", ppm=raw_ppm, contrast=contrast)

    # 3. Segmentation (TZ §5.5) and the calibration sources: pulses, header OCR (§5.4).
    c_cfg = cfg["calibration"]
    mask = make_segmenter(cfg.get("segmentation", {})).segment(rect.bgr, ppm)
    ink = mask.ink
    pulses = find_pulses(ink, ppm)
    dbg.pulses = pulses
    t = tick("segment+pulse", t)

    # 4. Layout: trace rows + registry; the page frame (rows, time origin, ink extent).
    guess = detect_layout(ink, ppm, pitch_tol=cfg["layout"]["pitch_tolerance"], pulse_bottoms=[p.bottom for p in pulses])
    dbg.rows = guess.rows
    if hints.layout:
        layout_name, layout_conf = hints.layout, 1.0
    else:
        layout_name, layout_conf = guess.name, guess.confidence
        if guess.uncertain and guess.name:
            warnings.append("LAYOUT_UNCERTAIN")
    frame = None
    if layout_name is not None:
        spec = get_layout(layout_name)
        frame = frame_page(spec, guess.baselines, long_stroke_mask(ink, ppm), ppm, [p.x_right for p in pulses])
    dbg.frame = frame
    if (frame is not None and not hints.layout and "LAYOUT_UNCERTAIN" in warnings
            and abs(frame.pitch_px / ppm - spec.row_pitch_mm) <= cfg["layout"]["pitch_tolerance"] * spec.row_pitch_mm):
        # Rows filled/picked to the registry count at the registry pitch: the layout is settled.
        warnings.remove("LAYOUT_UNCERTAIN")
        layout_conf = max(layout_conf, 0.8)
    t = tick("layout", t)
    if layout_name is None or frame is None:
        reasons.append("LAYOUT_UNKNOWN")
        return result("rejected", ppm=raw_ppm, contrast=contrast,
                      layout=Layout(name=layout_name, confidence=layout_conf))
    # Header / footer OCR only outside the trace rows (TZ §5.4: never on the trace).
    ocr_speeds: list[float] = []
    ocr_gains: list[float] = []
    if c_cfg["ocr"]["enabled"]:
        h_img = rect.bgr.shape[0]
        top = int(max(0.0, frame.baselines[0] - 0.6 * frame.pitch_px))
        bottom = int(min(h_img, frame.baselines[-1] + 0.6 * frame.pitch_px))
        reading = read_bands(rect.bgr, header_frac=top / h_img, footer_frac=(h_img - bottom) / h_img,
                             min_band_px=int(4 * ppm))
        dbg.ocr_texts = reading.texts
        ocr_speeds, ocr_gains = reading.speeds, reading.gains
    t = tick("ocr", t)
    decision = resolve(px_per_mm=ppm, grid_residual_mm=rect.residual_mm, pulse=summarize_pulses(pulses),
                       ocr_speeds=ocr_speeds, ocr_gains=ocr_gains, cfg=c_cfg,
                       pulse_width_s=spec.pulse.width_s if spec.pulse.present else None,
                       layout_speed=frame.implied_speed_mm_s if layout_conf >= 0.7 else None,
                       user_speed=hints.paper_speed_mm_s, user_gain=hints.gain_mm_mv)
    decision.calibration.px_per_mm_x = round(raw.px_per_mm_x, 4) if raw.px_per_mm_x else None
    decision.calibration.px_per_mm_y = round(raw.px_per_mm_y, 4) if raw.px_per_mm_y else None
    warnings += decision.warnings

    # 5. Traces: registry panels, vectorisation, validation (TZ §5.7-5.8).
    v_cfg = cfg.get("vectorize", {})
    try:
        readings = read_panels(mask, panels(spec, frame, ppm, decision.speed_for_traces), px_per_mm=ppm,
                               speed_mm_s=decision.speed_for_traces, gain_mm_mv=decision.gain_for_traces,
                               pitch_px=frame.pitch_px, fs_out=fs_out, cfg=v_cfg)
    except Exception:
        reasons.append("PROCESSING_ERROR")
        return result("rejected", calibration=decision.calibration, ppm=raw_ppm, contrast=contrast)
    dbg.readings = readings
    t = tick("traces", t)
    leads = {r.panel.lead: r.trace for r in readings if r.panel.kind == "panel"}
    rhythm = [r.trace for r in readings if r.panel.kind == "rhythm"]
    val = cfg.get("validate", {})
    cover = [r.trace.coverage or 0.0 for r in readings]
    if float(np.median(cover)) < val.get("min_median_coverage", 0.5):
        reasons.append("TRACE_BROKEN")
        return result("rejected", calibration=decision.calibration, ppm=raw_ppm, contrast=contrast,
                      layout=Layout(name=layout_name, confidence=layout_conf))
    cons = limb_consistency([(r.panel.lead, r.trace) for r in readings], fs_out)
    dbg.consistency = cons
    if cons.score is not None and cons.score < val.get("limb_warn", 0.7):
        warnings.append("LIMB_LEADS_INCONSISTENT")
    if any("TRACE_GAPS" in r.trace.flags for r in readings):
        warnings.append("TRACE_GAPS")
    if any("LEAD_OVERLAP" in r.trace.flags for r in readings):
        warnings.append("LEAD_OVERLAP")
    if min(cover) < val.get("lead_min_coverage", 0.8):
        warnings.append("LEAD_INCOMPLETE")          # a lead is largely missing (glare, stain): the doctor looks
    lead_q = float(np.median([r.quality for r in readings]))
    cons_f = 1.0 if cons.score is None else float(np.clip(0.5 + cons.score, 0.0, 1.0))
    overall = lead_q * cons_f * raw.confidence ** 0.5 * (decision.calibration.confidence if decision.established else 0.5)
    trusted = (decision.established and overall >= val.get("ok_min_overall", 0.6)
               and not {"LAYOUT_UNCERTAIN", "LIMB_LEADS_INCONSISTENT", "LEAD_INCOMPLETE"} & set(warnings))
    status = "ok" if trusted else "needs_confirmation"
    layout = Layout(name=layout_name, registry_version=spec.registry_version, confidence=layout_conf)
    res = result(status, calibration=decision.calibration, layout=layout, leads=leads, rhythm=rhythm,
                 overall=overall, ppm=raw_ppm, contrast=contrast)
    res.quality.limb_lead_consistency = cons.score
    if status == "ok" and val.get("beat_check", True) and beats_agree(res) is False:
        # Accurate traces can still give a wrong rhythm downstream (CardioLens D92): the
        # beats the measurement keeps disagree with the beats the page columns show.
        res.warnings = sorted({*res.warnings, "BEATS_INCONSISTENT"})
        res.status = "needs_confirmation"
    t = tick("validate", t)
    if artifacts_dir is not None:
        from ecgscan.artifacts import write_artifacts

        res.artifacts = write_artifacts(artifacts_dir, rectified=rect.bgr, mask=mask.ink, readings=readings)
        tick("artifacts", t)
    return res
