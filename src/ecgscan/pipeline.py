"""ecgscan pipeline: image -> ScanResult (TZ §5).

Intake, grid and resolution check, rectification to a known px/mm, calibration
from pulse + header OCR (M2); segmentation, registry panels, vectorisation and
physiological validation (M3).

Status rules:
* ``rejected`` — LOW_RESOLUTION, GRID_NOT_FOUND, LOW_CONTRAST, LAYOUT_UNKNOWN,
  TRACE_BROKEN, PROCESSING_ERROR (no traces are returned);
* ``needs_confirmation`` — speed or gain not established (SD-011), sources disagree,
  layout uncertain, limb leads inconsistent, a lead largely missing, the CardioLens
  beats disagreeing with the page columns (D130), or overall quality below
  ``validate.ok_min_overall``;
* ``ok`` — otherwise.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecgscan import PIPELINE_VERSION
from ecgscan.board import compose_zone, detect_board, load_boards, warp_zone
from ecgscan.calibration.header_ocr import read_bands
from ecgscan.calibration.pulse import find_pulses, summarize_pulses, widen_strokes
from ecgscan.calibration.resolve import resolve
from ecgscan.config import load_yaml
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source
from ecgscan.contract.models import RHYTHM_MISMATCH
from ecgscan.geometry.grid import GridEstimate, estimate_grid, estimate_grid_tiled
from ecgscan.geometry.rectify import Rectified, rectify, tile_residual
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
    board: object | None = None             # BoardFit (a list for a set of parts), None: no marker seen
    composite: object | None = None         # board.Composite of a set of parts
    timings: dict[str, float] = field(default_factory=dict)


_QC_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ecgscan-qc")


_BOARDS: dict[tuple[str, ...], list] = {}


def _fits(works: list[np.ndarray], b_cfg: dict, *, whole_zone: bool) -> list:
    key = tuple(b_cfg.get("specs", []))
    if key not in _BOARDS:
        _BOARDS[key] = load_boards(list(key))
    return [detect_board(w, _BOARDS[key], min_markers=int(b_cfg.get("min_markers", 3)),
                         max_residual_mm=float(b_cfg.get("max_residual_mm", 1.0)),
                         max_side_px=int(b_cfg.get("detect_max_side_px", 2000)), whole_zone=whole_zone)
            for w in works]


def _zone_rect(zone: np.ndarray, raw: GridEstimate, wppm: float, name: str, cfg: dict,
               *, rescale_to_grid: bool) -> Rectified | None:
    """The rectified zone, cross-checked with the paper grid. The paper is the ECG's own
    scale: a page printed scaled or lifted off the board disagrees with the board by more
    than ``grid_mismatch``. One photo: ``None`` (the grid path reads it instead); a set of
    parts (no grid path for them): the zone is rescaled so the grid is exact."""
    import cv2

    grid = estimate_grid(zone)
    if (grid.px_per_mm is not None and grid.confidence >= cfg["quality"]["min_grid_confidence"]
            and abs(grid.px_per_mm / wppm - 1) > float(cfg["board"].get("grid_mismatch", 0.03))):
        if not rescale_to_grid:
            return None
        zone = cv2.resize(zone, None, fx=wppm / grid.px_per_mm_x, fy=wppm / grid.px_per_mm_y,
                          interpolation=cv2.INTER_AREA)
        grid = estimate_grid(zone)
    residual, pitches = tile_residual(zone, wppm)
    return Rectified(zone, cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY), wppm, grid, grid, residual, pitches,
                     method=f"board:{name}")


def _board(work: np.ndarray, cfg: dict, dbg: ScanDebug) -> tuple[Rectified | None, GridEstimate, float] | None:
    """The zone of a capture board in the photo rectified at the working scale, the
    photo's scale as a grid estimate (the board's mean) and its worst px/mm over the
    zone; ``None``: no usable board (the reason is in ``dbg.board``). Below the
    resolution floor the rectified image is ``None``: nothing is warped."""
    b_cfg = cfg.get("board") or {}
    if not b_cfg.get("enabled"):
        return None
    fit = _fits([work], b_cfg, whole_zone=True)[0]
    dbg.board = fit
    if fit is None or not fit.ok:
        return None
    wppm = float(cfg["geometry"]["working_px_per_mm"])
    raw = GridEstimate(fit.mean_px_per_mm, fit.mean_px_per_mm, 0.0, 0.0, "board", 1.0)
    if fit.min_px_per_mm < cfg["quality"]["min_px_per_mm"]:
        return None, raw, fit.min_px_per_mm
    rect = _zone_rect(warp_zone(work, fit, wppm), raw, wppm, fit.spec.name, cfg, rescale_to_grid=False)
    if rect is None:
        fit.ok = False
        fit.reason = "the paper grid disagrees with the board scale"
        return None
    return rect, raw, fit.min_px_per_mm


def _board_set(works: list[np.ndarray], cfg: dict, dbg: ScanDebug) -> tuple[Rectified | None, GridEstimate, float] | str:
    """A page photographed in parts on one board (SD-066): the parts composed into one
    zone, or the refusal code. Every part needs its own markers; the composed zone must
    be covered (``board.min_coverage``) and is judged at its worst best-resolution."""
    b_cfg = cfg.get("board") or {}
    if not b_cfg.get("enabled"):
        return "BOARD_NOT_FOUND"
    fits = _fits(works, b_cfg, whole_zone=False)
    dbg.board = fits
    if any(f is None or not f.ok for f in fits) or len({f.spec.name for f in fits}) != 1:
        return "BOARD_NOT_FOUND"
    wppm = float(cfg["geometry"]["working_px_per_mm"])
    comp = compose_zone(works, fits, wppm)
    dbg.composite = comp
    if comp.coverage < float(b_cfg.get("min_coverage", 0.99)):
        return "PAGE_INCOMPLETE"
    raw = GridEstimate(comp.mean_px_per_mm, comp.mean_px_per_mm, 0.0, 0.0, "board", 1.0)
    if comp.min_px_per_mm < cfg["quality"]["min_px_per_mm"]:
        return None, raw, comp.min_px_per_mm
    return _zone_rect(comp.bgr, raw, wppm, fits[0].spec.name, cfg, rescale_to_grid=True), raw, comp.min_px_per_mm


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


def scan(image: str | Path | bytes | list[str | Path | bytes], *, hints: ScanHints | None = None, fs_out: float = 500.0,
         debug: ScanDebug | None = None, artifacts_dir: str | Path | None = None,
         config: dict | None = None) -> ScanResult:
    """Digitise one photo/scan. ``artifacts_dir``: write rectified / mask / overlay images there.

    A list: one page photographed in parts on a capture board (SD-066); the parts are
    composed by the board's markers, and a part without them refuses the set."""
    t_start = time.perf_counter()
    cfg = config or pipeline_config()
    hints = hints or ScanHints()
    dbg = debug if debug is not None else ScanDebug()
    q_cfg, g_cfg = cfg["quality"], cfg["geometry"]

    def tick(name: str, t0: float) -> float:
        dbg.timings[name] = round(time.perf_counter() - t0, 3)
        return time.perf_counter()

    parts = list(image) if isinstance(image, (list, tuple)) else [image]
    if not parts:
        raise ValueError("scan: no image")
    imgs = []
    try:
        for part in parts:
            imgs.append(load_image(part, max_megapixels=float(q_cfg.get("max_megapixels", 60)),
                                   min_side_px=int(q_cfg.get("min_side_px", 200))))
    except ImageRefused as exc:
        # Refused before decoding (D136): nothing of the picture was looked at.
        data = part if isinstance(part, bytes) else Path(part).read_bytes()
        return ScanResult(
            scan_id=str(uuid.uuid4()), status="rejected", reject_reasons=[exc.code],
            source=Source(modality="photo", original_sha256=hashlib.sha256(data).hexdigest(),
                          original_bytes=len(data), mime_type=_mime(data), client=hints.client,
                          device=hints.device, header_masked=hints.header_masked, pipeline_version=PIPELINE_VERSION),
            calibration=Calibration(confidence=0.0), layout=Layout(confidence=0.0), sampling_rate_hz=fs_out,
            quality=QualitySummary(overall=0.0, processing_ms=int((time.perf_counter() - t_start) * 1000)))
    img = imgs[0]
    source = Source(modality="photo", original_sha256=img.original_sha256, original_bytes=img.original_bytes,
                    mime_type=img.mime_type, width_px=img.width_px, height_px=img.height_px,
                    exif_orientation=img.exif_orientation, client=hints.client, device=hints.device,
                    header_masked=hints.header_masked, pipeline_version=PIPELINE_VERSION)
    if len(imgs) > 1:
        shas = [i.original_sha256 for i in imgs]
        source.parts_sha256 = shas
        source.original_sha256 = hashlib.sha256("".join(shas).encode()).hexdigest()
        source.original_bytes = sum(i.original_bytes for i in imgs)
    reasons: list[str] = []
    warnings: list[str] = []
    # The phone's frame checks, repeated on the photo as received (TZ §6.1, SD-046):
    # recorded for the record and for comparing client and server, not a gate.
    frame_qc = None
    qc_future = None
    if q_cfg.get("capture_qc", True):
        # Recorded only, never a gate: computed alongside the rest (~1 s; SD-063).
        from ecgscan.capture_qc import check_frame

        qc_future = _QC_POOL.submit(check_frame, img.bgr[..., ::-1], grid=False)

    def result(status: str, *, calibration: Calibration | None = None, layout: Layout | None = None,
               leads=None, rhythm=None, overall: float = 0.0, ppm: float | None = None,
               contrast: float | None = None) -> ScanResult:
        nonlocal frame_qc
        if qc_future is not None and frame_qc is None:
            frame_qc = qc_future.result()
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
    works = []
    for im in imgs:
        works.append(im.bgr)
        if paper_level(im.bgr) < q_cfg.get("flatfield_below", 0):
            works[-1] = flat_field(im.bgr, target=float(q_cfg.get("flatfield_target", 235)))
            warnings.append("LOW_LIGHT")
    work = works[0]

    # 0b. A capture board (SD-065): its markers give the exact scale and perspective and the
    # photo's resolution over the whole zone; the scan then sees only the zone. No board, or
    # one that cannot be trusted: the grid path below, unchanged.
    t = time.perf_counter()
    if len(works) > 1:
        on_board = _board_set(works, cfg, dbg)
        if isinstance(on_board, str):
            reasons.append(on_board)
            return result("rejected")
        warnings.append("STITCHED")
    else:
        on_board = _board(work, cfg, dbg)
    t = tick("board", t)
    if on_board is not None:
        rect, raw, raw_ppm = on_board
        if rect is None:
            reasons.append("LOW_RESOLUTION")
            return result("rejected", ppm=raw_ppm)
        if raw_ppm < q_cfg["marginal_px_per_mm"]:
            warnings.append("LOW_RESOLUTION_MARGINAL")
        dbg.rectified = rect
    else:
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
                           shear_min_deg=g_cfg.get("shear_min_deg"),
                           reread_tolerance=float(g_cfg.get("reread_tolerance", 0.15)))
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
    if contrast < q_cfg.get("faded_below", 0.0):
        # Pale but readable (faded thermal paper): read, flagged, and never "ok" by itself
        # (SD-064: 4 of 6 synthetic faded pages read at r 0.97-0.99; the 2 misread ones
        # were flagged by the limb / rhythm checks).
        warnings.append("FADED_PAPER")

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
        # The ink ends well before the layout's last column: the photo shows part of the
        # page (a piece of a strip), not broken traces. Only the reason differs; both refuse,
        # and a part of a page is a layout matter for the callers (SD-067).
        width_px = spec.record_s * decision.speed_for_traces * ppm
        shown = None if frame.ink_x1 is None or width_px <= 0 else (frame.ink_x1 - frame.x0) / width_px
        partial = shown is not None and shown < val.get("min_page_extent", 0.8)
        reasons.append("PAGE_INCOMPLETE" if partial else "TRACE_BROKEN")
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
               and not ("LOW_LIGHT" in warnings and q_cfg.get("low_light_needs_confirmation", True))
               and "FADED_PAPER" not in warnings)
    status = "ok" if trusted else "needs_confirmation"
    layout = Layout(name=layout_name, registry_version=spec.registry_version, confidence=layout_conf)
    res = result(status, calibration=decision.calibration, layout=layout, leads=leads, rhythm=rhythm,
                 overall=overall, ppm=raw_ppm, contrast=contrast)
    res.quality.limb_lead_consistency = cons.score
    hump_scale = val.get("qrs_hump_scale")
    if status == "ok" and (val.get("beat_check", True) or hump_scale):
        check = measurement_check(res, hump_scale=hump_scale)
        if val.get("beat_check", True) and check.beats_agree is False:
            # Accurate traces can still give a wrong rhythm downstream (CardioLens D130): the
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
        # image the service keeps or sends, whatever the client masked (D136).
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
