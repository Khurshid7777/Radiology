"""ecgscan pipeline: image -> ScanResult (TZ §5).

M2 scope: intake, grid and resolution check, rectification to a known px/mm,
calibration from pulse + header OCR with honest statuses. Layout and traces come
from the CardioLens reader via ``ecgscan.legacy`` until M3.

Status rules:
* ``rejected`` — LOW_RESOLUTION, GRID_NOT_FOUND, LOW_CONTRAST, LAYOUT_UNKNOWN,
  TRACE_BROKEN, PROCESSING_ERROR (no traces are returned);
* ``needs_confirmation`` — speed or gain not established (SD-011), or sources disagree;
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
from ecgscan.calibration.pulse import find_pulses, ink_mask, summarize_pulses
from ecgscan.calibration.resolve import resolve
from ecgscan.config import load_yaml
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source
from ecgscan.geometry.grid import estimate_grid
from ecgscan.geometry.rectify import Rectified, rectify
from ecgscan.io import load_image
from ecgscan.layout import get_layout
from ecgscan.layout.detect import detect_layout

TRACE_READERS = ("3x4+rhythm", "12x1")   # layouts the M2 (legacy) trace reader handles


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
         debug: ScanDebug | None = None) -> ScanResult:
    t_start = time.perf_counter()
    cfg = pipeline_config()
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

    # 3. Calibration: pulses, header OCR, resolution of the sources (TZ §5.4).
    c_cfg = cfg["calibration"]
    ink = ink_mask(rect.bgr, ppm)
    pulses = find_pulses(ink, ppm)
    dbg.pulses = pulses
    t = tick("pulse", t)
    ocr_speeds: list[float] = []
    ocr_gains: list[float] = []
    if c_cfg["ocr"]["enabled"]:
        reading = read_bands(rect.bgr, header_frac=c_cfg["ocr"]["header_frac"], footer_frac=c_cfg["ocr"]["footer_frac"])
        dbg.ocr_texts = reading.texts
        ocr_speeds, ocr_gains = reading.speeds, reading.gains
    t = tick("ocr", t)

    # 4. Layout: trace rows + registry (M3 adds lead labels).
    if hints.layout:
        layout_name, layout_conf = hints.layout, 1.0
    else:
        guess = detect_layout(ink, ppm, pitch_tol=cfg["layout"]["pitch_tolerance"])
        dbg.rows = guess.rows
        layout_name, layout_conf = guess.name, guess.confidence
        if guess.uncertain and guess.name:
            warnings.append("LAYOUT_UNCERTAIN")
    t = tick("layout", t)
    if layout_name is None or layout_name not in TRACE_READERS:
        # 6x2 / single are recognised but the M2 trace reader cannot read them (M3): refuse, do not guess.
        reasons.append("LAYOUT_UNKNOWN")
        return result("rejected", ppm=raw_ppm, contrast=contrast,
                      layout=Layout(name=layout_name, confidence=layout_conf))
    spec = get_layout(layout_name)
    decision = resolve(px_per_mm=ppm, grid_residual_mm=rect.residual_mm, pulse=summarize_pulses(pulses),
                       ocr_speeds=ocr_speeds, ocr_gains=ocr_gains, cfg=c_cfg,
                       pulse_width_s=spec.pulse.width_s if spec.pulse.present else None,
                       user_speed=hints.paper_speed_mm_s, user_gain=hints.gain_mm_mv)
    decision.calibration.px_per_mm_x = round(raw.px_per_mm_x, 4) if raw.px_per_mm_x else None
    decision.calibration.px_per_mm_y = round(raw.px_per_mm_y, 4) if raw.px_per_mm_y else None
    warnings += decision.warnings

    # 5. Traces (legacy reader on the rectified, calibrated page).
    from ecgscan.legacy import read_traces, signal_traces

    try:
        sig = read_traces(rect.gray, layout=layout_name, px_per_mm=ppm, speed_mm_s=decision.speed_for_traces,
                          gain_mm_mv=decision.gain_for_traces)
    except ValueError as exc:
        reasons.append("TRACE_BROKEN" if "trace" in str(exc) or "rows" in str(exc) else "PROCESSING_ERROR")
        return result("rejected", calibration=decision.calibration, ppm=raw_ppm, contrast=contrast)
    except Exception:
        reasons.append("PROCESSING_ERROR")
        return result("rejected", calibration=decision.calibration, ppm=raw_ppm, contrast=contrast)
    tick("traces", t)
    coverage = float(sig.provenance.confidence or 0.0)
    leads, rhythm = signal_traces(sig, fs_out=fs_out, quality=coverage)
    status = "ok" if decision.established and not ({"LAYOUT_UNCERTAIN"} & set(warnings)) else "needs_confirmation"
    overall = coverage * raw.confidence * (decision.calibration.confidence if decision.established else 0.5)
    layout = Layout(name=layout_name, registry_version=spec.registry_version, confidence=layout_conf)
    return result(status, calibration=decision.calibration, layout=layout, leads=leads, rhythm=rhythm,
                  overall=overall, ppm=raw_ppm, contrast=contrast)

