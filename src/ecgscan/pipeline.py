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

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecgscan import PIPELINE_VERSION
from ecgscan.calibration.header_ocr import read_bands
from ecgscan.calibration.pulse import find_pulses, summarize_pulses, widen_strokes
from ecgscan.calibration.resolve import resolve
from ecgscan.config import load_yaml
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source
from ecgscan.contract.models import RHYTHM_MISMATCH
from ecgscan.geometry.grid import estimate_grid, estimate_grid_tiled
from ecgscan.geometry.rectify import Rectified, rectify
from ecgscan.io import ImageRefused, _mime, flat_field, load_image, paper_level
from ecgscan.layout import get_layout
from ecgscan.layout.detect import detect_layout
from ecgscan.layout.panels import frame_page, panels
from ecgscan.layout.rows import long_stroke_mask
from ecgscan.segmentation import make_segmenter
from ecgscan.validate import limb_consistency, measurement_check, rhythm_check
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
    rhythm_checks: dict[str, dict[str, float]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def pipeline_config() -> dict:
    return load_yaml("pipeline.yaml")


def trace_contrast(bgr: np.ndarray, px_per_mm: float) -> float:
    """1 - (darkest neutral ink / paper): ~0.9 for a fresh print, low for faded thermal paper."""
    import cv2

    # uint8 throughout (the same percentiles, ~3x faster than on float copies; SD-060).
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, gr, r = cv2.split(bgr)
    chroma = cv2.subtract(cv2.max(cv2.max(b, gr), r), cv2.min(cv2.min(b, gr), r))
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

    try:
        img = load_image(image, max_megapixels=float(q_cfg.get("max_megapixels", 60)),
                         min_side_px=int(q_cfg.get("min_side_px", 200)))
    except ImageRefused as exc:
        # Refused before decoding (D98): nothing of the picture was looked at.
        data = image if isinstance(image, bytes) else Path(image).read_bytes()
        return ScanResult(
            scan_id=str(uuid.uuid4()), status="rejected", reject_reasons=[exc.code],
            source=Source(modality="photo", original_sha256=hashlib.sha256(data).hexdigest(),
                          original_bytes=len(data), mime_type=_mime(data), client=hints.client,
                          device=hints.device, header_masked=hints.header_masked, pipeline_version=PIPELINE_VERSION),
            calibration=Calibration(confidence=0.0), layout=Layout(confidence=0.0), sampling_rate_hz=fs_out,
            quality=QualitySummary(overall=0.0, processing_ms=int((time.perf_counter() - t_start) * 1000)))
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

    # 0. A dark photo (indoor light): the working copy is divided by the local paper level so
    # the segmenter, trained on lit pages, sees paper as paper (SD-062). Deterministic
    # photometry of the copy only; the original is hashed and kept as received (TZ §10).
    work = img.bgr
    paper = paper_level(img.bgr)
    if paper < q_cfg.get("flatfield_below", 0):
        work = flat_field(img.bgr, target=float(q_cfg.get("flatfield_target", 235)))
        warnings.append("LOW_LIGHT")

    # 1. Grid and resolution on the photo as taken (TZ §5.3).
    t = time.perf_counter()
    raw = estimate_grid(work)
    if work is not img.bgr:
        # Flattening can weaken a faint grid as well: the better of the two reads is kept.
        plain = estimate_grid(img.bgr)
        if plain.confidence > raw.confidence:
            raw = plain
    h_img, w_img = work.shape[:2]
    if raw.confidence < q_cfg["min_grid_confidence"] and max(h_img, w_img) > g_cfg.get("tile_aspect", 2.0) * min(h_img, w_img):
        # A long strip: perspective smears the whole-frame comb; read it piece by piece (SD-059).
        tiled = estimate_grid_tiled(work, tiles=int(g_cfg.get("grid_tiles", 3)))
        if tiled is not None and tiled.confidence > raw.confidence:
            raw = tiled
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
        rect = rectify(work, working_px_per_mm=float(g_cfg["working_px_per_mm"]), raw_grid=raw,
                       max_upscale=float(g_cfg["max_upscale"]), fit_px_per_mm=g_cfg.get("fit_px_per_mm"),
                       shear_min_deg=g_cfg.get("shear_min_deg"))
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
    p_cfg = c_cfg.get("pulse", {})
    if not pulses and p_cfg.get("retry_widen_px"):
        # A real strip's pulse: a thin line, slanted by the paper's curl, broken by the mask
        # into short pieces over two or three columns. Found again on a gap-tolerant copy;
        # only when nothing was found, so pages that already worked are unchanged (SD-059).
        pulses = find_pulses(widen_strokes(ink, int(p_cfg["retry_widen_px"]), int(p_cfg.get("retry_close_px", 5))), ppm)
    dbg.pulses = pulses
    t = tick("segment+pulse", t)

    # 4. Layout: trace rows + registry; the page frame (rows, time origin, ink extent).
    stroke_h = cfg["layout"].get("stroke_min_height_mm", 3.0)
    bridge = cfg["layout"].get("stroke_bridge_mm", 0.0)
    m2 = int(round(2.0 * ppm))       # a pulse's feet: about 1 mm on each side
    pulse_boxes = [(p.x_left - m2, p.top - m2, p.x_right + m2, p.bottom + m2) for p in pulses]
    guess = detect_layout(ink, ppm, pitch_tol=cfg["layout"]["pitch_tolerance"], pulse_bottoms=[p.bottom for p in pulses],
                          stroke_min_height_mm=stroke_h, stroke_bridge_mm=bridge, exclude_boxes=pulse_boxes)
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
        frame = frame_page(spec, guess.baselines, long_stroke_mask(ink, ppm, min_height_mm=stroke_h, bridge_mm=bridge,
                                                            exclude_boxes=pulse_boxes), ppm,
                           [p.x_right for p in pulses])
    dbg.frame = frame
    if (frame is not None and not hints.layout and "LAYOUT_UNCERTAIN" in warnings
            and abs(frame.pitch_px / ppm - spec.row_pitch_mm) <= cfg["layout"]["pitch_tolerance"] * spec.row_pitch_mm):
        # Rows filled/picked to the registry count at the registry pitch: the layout is settled.
        warnings.remove("LAYOUT_UNCERTAIN")
        layout_conf = max(layout_conf, 0.8)
    t = tick("layout", t)
    if (frame is not None and spec.start_pulse_required
            and (frame.column_x0 is None or not any(p.x_right <= frame.x0 + ppm for p in pulses))):
        # A strip names its groups by position from the start pulse: without the pulse, or
        # with groups missing, the photo does not show the whole strip (SD-059).
        reasons.append("PAGE_INCOMPLETE")
        return result("rejected", ppm=raw_ppm, contrast=contrast,
                      layout=Layout(name=layout_name, confidence=layout_conf))
    if layout_name is None or frame is None:
        reasons.append("LAYOUT_UNKNOWN")
        return result("rejected", ppm=raw_ppm, contrast=contrast,
                      layout=Layout(name=layout_name, confidence=layout_conf))
    # Header / footer OCR only outside the trace rows (TZ §5.4: never on the trace).
    ocr_speeds: list[float] = []
    ocr_gains: list[float] = []
    h_img = rect.bgr.shape[0]
    top = int(max(0.0, frame.baselines[0] - 0.6 * frame.pitch_px))
    bottom = int(min(h_img, frame.baselines[-1] + 0.6 * frame.pitch_px))
    if c_cfg["ocr"]["enabled"]:
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
    # The rhythm strip against the panels recorded with it (SD-060). Disagreement means one
    # of them is misread: the doctor looks. A strip unrelated to every panel it overlaps
    # (glare, a stain on the strip) stays in the result for the doctor but is flagged and
    # left out of the measurements: it gave HR 49 for 64 and 56 for 78.
    for rt in rhythm:
        checks = rhythm_check(leads, rt, fs_out)
        dbg.rhythm_checks[rt.lead] = checks
        if checks and max(checks.values()) > val.get("rhythm_warn", 0.5):
            warnings.append("RHYTHM_INCONSISTENT")
        if checks and min(checks.values()) >= val.get("rhythm_unrelated", 0.9):
            rt.flags = sorted({*rt.flags, RHYTHM_MISMATCH})
            rt.quality = 0.0
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
               and not {"LAYOUT_UNCERTAIN", "LIMB_LEADS_INCONSISTENT", "LEAD_INCOMPLETE",
                        "RHYTHM_INCONSISTENT"} & set(warnings)
               # Light correction is verified on synthetic pages only: the doctor confirms (SD-062).
               and not ("LOW_LIGHT" in warnings and q_cfg.get("low_light_needs_confirmation", True)))
    status = "ok" if trusted else "needs_confirmation"
    layout = Layout(name=layout_name, registry_version=spec.registry_version, confidence=layout_conf)
    res = result(status, calibration=decision.calibration, layout=layout, leads=leads, rhythm=rhythm,
                 overall=overall, ppm=raw_ppm, contrast=contrast)
    res.quality.limb_lead_consistency = cons.score
    hump_scale = val.get("qrs_hump_scale")
    if status == "ok" and (val.get("beat_check", True) or hump_scale):
        check = measurement_check(res, hump_scale=hump_scale)
        if val.get("beat_check", True) and check.beats_agree is False:
            # Accurate traces can still give a wrong rhythm downstream (CardioLens D92): the
            # beats the measurement keeps disagree with the beats the page columns show.
            res.warnings = sorted({*res.warnings, "BEATS_INCONSISTENT"})
            res.status = "needs_confirmation"
        if (check.qrs_ms is not None and check.qrs_ms_low_hump is not None
                and abs(check.qrs_ms_low_hump - check.qrs_ms) > val.get("qrs_unstable_ms", 20.0)):
            # The QRS end hinges on a small terminal R' (SD-057): exact traces, unstable width.
            res.warnings = sorted({*res.warnings, "QRS_UNSTABLE"})
            res.status = "needs_confirmation"
    t = tick("validate", t)
    if artifacts_dir is not None:
        from ecgscan.artifacts import write_artifacts

        # The header and footer bands carry the patient's name, ID and date: blanked in every
        # image the service keeps or sends, whatever the client masked (D98).
        bands = (top, bottom) if cfg.get("artifacts", {}).get("mask_bands", True) else None
        res.artifacts = write_artifacts(artifacts_dir, rectified=rect.bgr, mask=mask.ink, readings=readings,
                                        keep_rows=bands)
        tick("artifacts", t)
    return res


def warm_up(config: dict | None = None) -> float:
    """Scan one synthetic page so the first real scan does not pay for loading the
    segmentation model, the OCR engine and the CardioLens modules (~5 s, SD-056).
    Returns the seconds spent; never raises (a broken install shows on the real scan)."""
    t = time.perf_counter()
    try:
        import cv2

        from ecgscan.bench.cases import bench_config
        from ecgscan.bench.render import RenderSpec, render_page
        from ecgscan.bench.sources import pick_preset, synthetic_signal

        rng = np.random.default_rng(0)
        sig = synthetic_signal(pick_preset(bench_config()["sources"]["synthetic"]["adult"], rng), rng, 0)
        img, _ = render_page(sig.leads_mv, sig.fs, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
        scan(cv2.imencode(".png", img)[1].tobytes(), config=config)
    except Exception as exc:  # warm-up only: a broken install shows on the real scan
        import logging

        logging.getLogger(__name__).warning("scanner warm-up failed: %s", type(exc).__name__)
    return time.perf_counter() - t
