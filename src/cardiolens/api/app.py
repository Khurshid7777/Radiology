"""FastAPI service exposing the CardioLens pipeline (TZ §14).

Endpoints:
  GET  /                         -> web UI
  GET  /health                   -> liveness + which LLM provider is active
  GET  /api/cases                -> available synthetic demo cases
  POST /api/analyze              -> run pipeline (synthetic case or file upload)
  GET  /api/analysis/{id}/report.pdf
  GET  /api/analysis/{id}/overlay.jpg   (a photo read by the scanner: the traces over the page, D163)
  GET  /api/analysis/{id}/waveform.svg
  GET  /api/analysis/{id}/graph.png
  GET  /api/analysis/{id}/conclusion?lang=ru|uz
  POST /api/analysis/{id}/share  -> a signed 7-day link to the PDF, no photo (D197)
  GET  /s/{token}                -> that PDF, without a key

Everything under /api/analyze and /api/analysis needs an API key (``X-API-Key``, the same
keys as /scans in ``$ECGSCAN_API_KEYS``); a key sees only its own analyses (D136). The web
UI, /health and the demo case list stay open.

Core layers (1-5) run locally; the Layer-6 narrative comes from a local
OpenAI-compatible model, from Claude, or from the deterministic stub (TZ §12).
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import os
import secrets
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.background import BackgroundTask

from cardiolens.api import service
from cardiolens.api.guard import RequestGuard
from cardiolens.calibration import TemperatureScaler, expected_calibration_error
from cardiolens.config import load_app_config
from cardiolens.feedback import (
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_QUALITY_PATH,
    Feedback,
    FeedbackStore,
    quality,
)
from cardiolens.layer6_interpret import provider_status
from cardiolens.outputs import (
    format_conclusion,
    render_causal_graph,
    render_waveform,
    result_to_dict,
    write_pdf,
)
from ecgscan.config import load_yaml
from ecgscan_service.router import actor
from ecgscan_service.router import router as scans_router

app = FastAPI(title="CardioVision", version="0.1.0")
# Key check and body cap before any multipart parsing (D136): CardioLens uploads are 64 MB at
# most; /scans takes up to max_parts photos of one page on the capture board, each up to its
# own limit (SD-066, D145), as the standalone scanner service does.
_scan_upload = load_yaml("service")["upload"]
_SCANS_MAX_BYTES = int(_scan_upload["max_mb"] * 1024 * 1024) * int(_scan_upload.get("max_parts", 1))
app.add_middleware(RequestGuard, max_body_bytes=max(service.MAX_UPLOAD_BYTES, _SCANS_MAX_BYTES) + 1024 * 1024)
# ECG scanner (photo/scan input, TZ scanner §4, SD-004): /scans, own API-key auth.
app.include_router(scans_router)

_STATIC = Path(__file__).parent / "static"

# One analysis at a time, off the event loop (D156): a photo through the scanner and a Claude
# narrative take tens of seconds; meanwhile /health, the page and the other clinic's
# requests are still answered, and two analyses never share the server's memory.
_ANALYSIS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")


async def _in_analysis_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ANALYSIS, functools.partial(fn, *args, **kwargs))


def _cleanup(tmp_dir: str) -> BackgroundTask:
    """Delete a temp dir after the FileResponse has been sent (no disk leak)."""
    return BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)


def _quality_path() -> str | None:
    """D190: the real-use quality log, next to the feedback log; empty turns it off."""
    return os.environ.get("CARDIOLENS_QUALITY_PATH", DEFAULT_QUALITY_PATH) or None


def _feedback_store() -> FeedbackStore:
    return FeedbackStore.at(os.environ.get("CARDIOLENS_FEEDBACK_PATH", DEFAULT_FEEDBACK_PATH))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Liveness plus the Layer-6 narrative source the next analysis will use (TZ §12) and
    whether the learnable layer's model is loaded (D167: rules-only when the file is absent)."""
    from cardiolens.pipeline import ml_status

    cfg = load_app_config()
    return {
        "status": "ok", "service": "cardiolens", "version": "0.1.0",
        "llm": provider_status(cfg.llm),
        "analyses_kept_on_disk": service.persist.enabled(),     # D199
        "ml": ml_status(cfg),
    }


# --- D175: the web camera's live frame check -------------------------------------------------
CAPTURE_FRAME_MAX_BYTES = 2_000_000


@app.post("/api/capture/check")
async def capture_check(frame: UploadFile = File(...), capture_width: int = Form(default=0),
                        parts: bool = Form(default=False), who: str = Depends(actor)) -> dict:
    """One preview frame from the page's camera, checked as the phone app checks it (the
    scanner's ``capture_qc``, SD-014/SD-070), with what is in view told apart (D177): a
    device screen (no paper grid to check — light, glare and blur only), a sheet or a strip
    (its size in mm from the paper grid), and whether it must be taken in parts — a sheet
    running off the frame, or one that already fills the frame but is still too coarse.
    ``capture_width``: the width of the photo the camera will take. Nothing is stored."""
    import cv2
    import numpy as np

    from cardiolens.layer1_ingestion.screen import is_device_screen
    from ecgscan.capture_qc import check_frame, config
    from ecgscan.i18n import strings

    data = await frame.read(CAPTURE_FRAME_MAX_BYTES + 1)
    if len(data) > CAPTURE_FRAME_MAX_BYTES:
        raise HTTPException(413, "frame too large")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "not an image")
    h, w = bgr.shape[:2]
    scale = (capture_width / w) if capture_width and capture_width >= w else 1.0
    loop = asyncio.get_running_loop()
    ru, uz = strings("ru"), strings("uz")      # D200: the page shows either
    say = lambda codes: [{"code": c, "ru": ru.get(f"qc.{c}", c), "uz": uz.get(f"qc.{c}", ru.get(f"qc.{c}", c))}
                         for c in codes]

    screen = await loop.run_in_executor(None, is_device_screen, bgr)
    if screen:
        qc = await loop.run_in_executor(None, functools.partial(
            check_frame, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale_to_capture=scale, grid=False))
        hints = [c for c in qc.hints if c in ("glare", "steady", "light")]
        return {"ok": not hints, "mode": "screen", "mode_ru": "Экран прибора: снимите весь экран, без бликов",
                "hints": say(hints), "ready_ru": ru.get("qc.ready", ""), "ready_uz": uz.get("qc.ready", ""), "px_per_mm": None,
                "tilt_deg": None, "sheet_area": None, "sheet_mm": None, "corners": None,
                "board": False, "suggest_parts": False, "frame": [w, h]}

    cfg = config()
    if parts:
        cfg = {**cfg, "thresholds": {**cfg["thresholds"], "require_whole_sheet": False}}
    qc = await loop.run_in_executor(None, functools.partial(
        check_frame, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale_to_capture=scale, cfg=cfg))
    fit = await loop.run_in_executor(None, _board_in_view, bgr)
    board = fit is not None
    hints = list(qc.hints)
    corners = None if qc.corners is None else [[x / scale, y / scale] for x, y in qc.corners]
    min_ppm = float(cfg["thresholds"]["min_px_per_mm"])
    ppm = qc.px_per_mm
    if board:
        # The board's markers give the scale exactly; a preview misses a pale or fine paper
        # grid (the owner's strip: 1.9 px/mm read for ~4.7). Resolution, framing and size
        # follow the board (D177).
        ppm = round(fit.min_px_per_mm * scale, 3)
        hints = [c for c in hints if c not in ("closer", "frame")]
        if ppm < min_ppm:
            hints.insert(0, "closer")
        if not fit.zone_in_frame:
            hints.append("frame")
        sheet_mm = await loop.run_in_executor(None, _sheet_mm_on_board, bgr, fit)
    else:
        sheet_mm = _sheet_mm(corners, qc.px_per_mm, scale)
    # From large to small (D177): a sheet that runs off the frame, or fills it and is still
    # under the floor, cannot be taken whole at the scale needed — it is taken in parts.
    cut = qc.sheet_whole is False and (qc.sheet_area or 0) >= 0.5
    too_big = ppm is not None and ppm < min_ppm and (qc.sheet_area or 0) >= 0.6
    suggest_parts = not parts and not board and (cut or too_big)
    if suggest_parts:
        hints = [c for c in hints if c not in ("frame", "closer")] + ["parts"]
    if _sheet_on_end(corners) and (qc.sheet_area or 0) >= 0.3:
        hints.insert(0, "turn")        # D180: the rows run down the frame — the phone is held upright
    mode, mode_ru = _paper_kind(sheet_mm)
    texts = say([c for c in hints if c not in ("parts", "turn")])
    if "turn" in hints:
        texts.insert(0, {"code": "turn", "ru": "Поверните телефон горизонтально: строки ЭКГ должны идти слева направо.",
                         "uz": "Telefonni gorizontal holatga burang: EKG qatorlari chapdan o'ngga borishi kerak."})
    if board and sheet_mm:
        # Tinted paper is read up to the markers' inner edges (SD-077), not only the frame.
        zw, zh = _read_size(fit.spec)
        if sheet_mm[0] > max(zw, zh) + 2 or sheet_mm[1] > min(zw, zh) + 2:
            hints.append("board_small")
            texts.append({"code": "board_small", "ru": _board_advice(sheet_mm, fit.spec.name)})
    if "parts" in hints:
        texts.append({"code": "parts", "ru": "Лист большой: снимите его частями, каждая часть с перекрытием "
                                           "соседней — режим «частями» включён.",
                      "uz": "Varaq katta: uni qismlarga bo'lib suratga oling, har bir qism qo'shnisini qoplab "
                            "tursin — «qismlarga» rejimi yoqildi."})
    return {
        "ok": not hints,
        "mode": mode, "mode_ru": mode_ru,
        "hints": texts,
        "ready_ru": ru.get("qc.ready", ""), "ready_uz": uz.get("qc.ready", ""),
        "px_per_mm": ppm,
        "tilt_deg": qc.tilt_deg,
        "sheet_area": qc.sheet_area,
        "sheet_mm": sheet_mm,
        "board_name": fit.spec.name if board else None,
        "corners": corners,
        "board": board,
        "suggest_parts": suggest_parts,
        "frame": [w, h],
    }


def _sheet_on_end(corners) -> bool:
    """The sheet stands on end in the frame: its long side runs down (the strip photographed
    with the phone upright; ECG rows run along the long side)."""
    if not corners:
        return False
    c = np.asarray(corners, float)
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return h > 1.3 * w


def _sheet_mm(corners, px_per_mm, scale) -> list[float] | None:
    """The sheet's long and short side in mm: its corners on the frame over the paper grid's
    px/mm on the frame (``px_per_mm`` is on the captured photo, ``scale`` times larger)."""
    if not corners or not px_per_mm:
        return None
    c = np.asarray(corners, float)
    ppm = px_per_mm / scale
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2 / ppm
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2 / ppm
    return _plausible([max(w, h), min(w, h)])


def _sheet_mm_on_board(bgr, fit, res: float = 2.0) -> list[float] | None:
    """The paper's size in board mm (D177): the board straightened by its markers at ``res``
    px/mm, the paper found as the tinted area (grid lines, pink or cream paper) on the white
    board — the printed rulers and markers are grey and black, not tinted."""
    import cv2

    bw, bh = fit.spec.size_mm
    S = np.diag([res, res, 1.0]) @ fit.H
    flat = cv2.warpPerspective(bgr, S, (int(bw * res), int(bh * res)))
    chroma = flat.max(axis=2).astype(np.int16) - flat.min(axis=2).astype(np.int16)
    mask = (chroma > 14).astype(np.uint8)
    k = max(3, int(3 * res))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    w, h = stats[i, cv2.CC_STAT_WIDTH] / res, stats[i, cv2.CC_STAT_HEIGHT] / res
    return _plausible([max(w, h), min(w, h)])


def _board_advice(sheet_mm, current: str) -> str:
    """Which capture board's zone holds a paper of ``sheet_mm`` (D177)."""
    from ecgscan.board import load_boards
    from ecgscan.pipeline import pipeline_config

    specs = load_boards(list((pipeline_config().get("board") or {}).get("specs", [])))
    sizes = [(s, _read_size(s)) for s in specs]
    fits = sorted((w * h, s.name, s) for s, (w, h) in sizes
                  if max(w, h) + 2 >= sheet_mm[0] and min(w, h) + 2 >= sheet_mm[1])
    names = {"A3x2": "A3x2 (A3 из двух листов A4)", "A4W": "A4W (A4 с широкой рамкой)"}
    name = fits[0][2].title.split()[-1] if fits else ""
    tail = f"подойдёт подложка {names.get(name, name)}" if fits else "снимите её частями или без подложки"
    return (f"Бумага {sheet_mm[0]} × {sheet_mm[1]} мм больше этой подложки ({current.upper()}) — "
            f"край будет обрезан; {tail}.")


def _read_size(spec) -> tuple[float, float]:
    """How much paper a board holds: up to its markers when the scanner reads beyond the
    frame (SD-077, tinted paper or a coloured grid), else the frame."""
    from ecgscan.pipeline import pipeline_config

    b_cfg = pipeline_config().get("board") or {}
    if b_cfg.get("read_beyond_zone") and hasattr(spec, "read_mm"):
        x0, y0, x1, y1 = spec.read_mm(float(b_cfg.get("read_margin_mm", 1.5)))
        return x1 - x0, y1 - y0
    return spec.zone_size_mm


def _plausible(sides) -> list[float] | None:
    """An ECG paper is 40-600 mm long and 25-320 mm wide; anything else is a misread."""
    long_, short = sides
    return [round(long_), round(short)] if 40 <= long_ <= 600 and 25 <= short <= 320 else None


def _paper_kind(sheet_mm) -> tuple[str, str]:
    if not sheet_mm:
        return "paper", "Бумажная ЭКГ"
    long_, short = sheet_mm
    kind = "strip" if short <= 160 else "sheet"
    word = "Лента" if kind == "strip" else "Лист"
    return kind, f"{word} ≈ {long_} × {short} мм"


def _board_in_view(bgr):
    """The capture board's fit when enough of its markers are in view and trusted, else None."""
    try:
        from ecgscan.board import detect_board, load_boards
        from ecgscan.pipeline import pipeline_config
    except ImportError:
        return None
    b_cfg = pipeline_config().get("board") or {}
    specs = load_boards(list(b_cfg.get("specs", [])))
    fit = detect_board(bgr, specs, min_markers=int(b_cfg.get("min_markers", 3)), whole_zone=False)
    return fit if fit is not None and fit.ok else None


@app.get("/api/cases")
def cases() -> dict:
    return {"cases": sorted(service.DEMO_CASES)}


def _allowed(v: float | None, values: tuple[float, ...]) -> float | None:
    """A paper speed or gain the recorders print, else None (auto)."""
    if v is None or v <= 0:
        return None
    if not any(abs(v - x) < 1e-6 for x in values):
        raise ValueError(f"unsupported value {v:g} (allowed: {', '.join(f'{x:g}' for x in values)})")
    return float(v)


@app.post("/api/analyze")
async def analyze(
    case: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    sampling_rate_hz: float | None = Form(default=None),
    unit: str = Form(default="uv"),
    age_years: float | None = Form(default=None),
    age_days: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    potassium: float | None = Form(default=None),
    calcium: float | None = Form(default=None),
    prior_ecg_id: str | None = Form(default=None),
    symptoms: str | None = Form(default=None),        # D195: "chest_pain,syncope"
    symptom_onset: str | None = Form(default=None),
    clinical_note: str | None = Form(default=None),   # D209: the doctor's diagnosis / description
    client: str | None = Form(default=None),     # "telegram": refusals keep the bot's wording (D161)
    paper_speed_mm_s: float | None = Form(default=None),   # D179: printed on the strip, given by the user
    gain_mm_mv: float | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
    who: str = Depends(actor),
) -> JSONResponse:
    try:
        if files:
            uploads, budget = [], service.MAX_UPLOAD_BYTES
            for f in files:
                # A running byte budget: never more than the limit in memory (D136).
                content = await f.read(budget + 1)
                budget -= len(content)
                if budget < 0:
                    raise ValueError(f"upload too large (> {service.MAX_UPLOAD_BYTES} bytes)")
                uploads.append(service.UploadFile(filename=f.filename or "upload", content=content))
            sig = await _in_analysis_thread(
                service.signal_from_uploads, uploads, sampling_rate_hz=sampling_rate_hz, unit=unit,
                paper_speed_mm_s=_allowed(paper_speed_mm_s, (12.5, 25.0, 50.0, 100.0)),
                gain_mm_mv=_allowed(gain_mm_mv, (2.5, 5.0, 10.0, 20.0, 40.0)),
                wording="bot" if client == "telegram" else "web",
            )
        elif case:
            sig = service.signal_from_case(case)
        else:
            raise ValueError("provide a 'case' or upload 'files'")

        service.apply_patient_overrides(
            sig, age_years=age_years, age_days=age_days, sex=sex,
            potassium=potassium, calcium=calcium,
            symptoms=[x.strip() for x in (symptoms or "").split(",") if x.strip()],
            symptom_onset=symptom_onset, clinical_note=clinical_note,
        )
        analysis_id, result = await _in_analysis_thread(
            service.run_and_store, sig, mode=mode, prior_id=prior_ecg_id, owner=who
        )
    except ValueError as exc:
        if files:
            quality.append(_quality_path(), quality.refusal_row(str(exc), client))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if files:
        quality.append(_quality_path(), quality.quality_row(result, client))
    payload = result_to_dict(result)
    payload["analysis_id"] = analysis_id
    payload["conclusion_ru"] = format_conclusion(result, "ru")
    payload["conclusion_uz"] = format_conclusion(result, "uz")
    # A photo the scanner read: the page shows what was read (D163), the bot sends it.
    payload["overlay_available"] = result.signal.provenance.overlay_jpeg is not None
    from cardiolens.outputs.pdf_report import device_line, speed_assumed_note

    payload["speed_note_ru"] = speed_assumed_note(result)       # D180
    payload["device_ru"] = device_line(result)                  # D183
    return JSONResponse(payload)


def _require(analysis_id: str, who: str):
    result = service.get_analysis(analysis_id, owner=who)
    if result is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return result


# D197: a shared link names the analysis and its expiry, signed with the server's secret. No patient
# data in it; without a configured secret one is made at start (links then end with a restart, as the
# in-memory analyses do).
SHARE_DAYS = 7
_SHARE_SECRET = (os.environ.get("CARDIOLENS_SHARE_SECRET") or "").encode() or secrets.token_bytes(32)


def _share_sig(analysis_id: str, exp: int) -> str:
    return hmac.new(_SHARE_SECRET, f"{analysis_id}.{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def share_token(analysis_id: str, exp: int) -> str:
    return f"{analysis_id}.{exp}.{_share_sig(analysis_id, exp)}"


def _share_target(token: str) -> str | None:
    """The analysis a token names, or None when it is forged, malformed or expired."""
    parts = token.split(".")
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    aid, exp, sig = parts[0], int(parts[1]), parts[2]
    if not hmac.compare_digest(sig, _share_sig(aid, exp)) or exp < time.time():
        return None
    return aid


@app.post("/api/analysis/{analysis_id}/share")
def share(analysis_id: str, who: str = Depends(actor)) -> JSONResponse:
    _require(analysis_id, who)                  # only the owner's analysis
    exp = int(time.time()) + SHARE_DAYS * 86400
    return JSONResponse({"url": f"/s/{share_token(analysis_id, exp)}", "expires_at": exp})


@app.get("/s/{token}")
def shared_pdf(token: str):
    aid = _share_target(token)
    if aid is None:
        return PlainTextResponse("Ссылка недействительна или её срок истёк.", status_code=404)
    result = service.get_analysis(aid)
    if result is None:
        return PlainTextResponse("Этот анализ больше не хранится на сервере. Попросите прислать PDF файлом.",
                                 status_code=410)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{aid}.pdf")
    write_pdf(result, path, include_photo=False)
    return FileResponse(path, media_type="application/pdf", filename=f"cardiolens_{aid}.pdf",
                        headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex"},
                        background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/report.pdf")
def report_pdf(analysis_id: str, who: str = Depends(actor)) -> FileResponse:
    result = _require(analysis_id, who)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.pdf")
    write_pdf(result, path)
    return FileResponse(path, media_type="application/pdf",
                        filename=f"cardiolens_{analysis_id}.pdf",
                        background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/overlay.jpg")
def overlay_jpg(analysis_id: str, who: str = Depends(actor)) -> Response:
    """The scanner's confirmation overlay of a photo (D163): the traces it read, drawn over
    the page, patient header blanked. 404 for a signal file, a synthetic case, or a photo
    the CardioLens readers read instead of the scanner."""
    result = _require(analysis_id, who)
    data = result.signal.provenance.overlay_jpeg
    if data is None:
        raise HTTPException(status_code=404, detail="no overlay for this analysis")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/analysis/{analysis_id}/waveform.svg")
def waveform_svg(analysis_id: str, who: str = Depends(actor)) -> FileResponse:
    result = _require(analysis_id, who)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.svg")
    render_waveform(result, path)
    return FileResponse(path, media_type="image/svg+xml", background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/graph.png")
def graph_png(analysis_id: str, who: str = Depends(actor)) -> FileResponse:
    result = _require(analysis_id, who)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.png")
    render_causal_graph(result, path)
    return FileResponse(path, media_type="image/png", background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/conclusion", response_class=PlainTextResponse)
def conclusion(analysis_id: str, lang: str = "ru", who: str = Depends(actor)) -> str:
    result = _require(analysis_id, who)
    return format_conclusion(result, "uz" if lang == "uz" else "ru")


@app.post("/api/analysis/{analysis_id}/feedback")
def submit_feedback(analysis_id: str, fb: Feedback, who: str = Depends(actor)) -> dict:
    """Capture physician corrections for future fine-tuning/calibration (TZ §13)."""
    fb.analysis_id = analysis_id
    result = _require(analysis_id, who)
    if result is not None and fb.mode is None:
        fb.mode = result.report.mode
    _feedback_store().append(fb)
    return {"status": "recorded", "analysis_id": analysis_id,
            "n_findings": len(fb.findings)}


@app.get("/api/quality/summary")
def quality_summary(days: int = 30, who: str = Depends(actor)) -> dict:
    """D190: aggregates of the real-use quality log (counts and errors only)."""
    import time as _time

    rows = quality.read(_quality_path())
    return quality.summarize(rows, since_ts=int(_time.time()) - max(1, days) * 86400)


@app.post("/api/calibration/fit")
def fit_calibration(who: str = Depends(actor)) -> dict:
    """Fit the confidence calibrator from accumulated feedback (TZ §13, §16)."""
    samples = _feedback_store().calibration_dataset()
    scaler = TemperatureScaler().fit(samples)
    ece_before = expected_calibration_error(samples)
    ece_after = expected_calibration_error([(scaler.apply(c), y) for c, y in samples])
    return {
        "n_samples": len(samples),
        "fitted": scaler.fitted,
        "temperature": round(scaler.temperature, 4),
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
