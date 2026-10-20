"""FastAPI service exposing the CardioLens pipeline (TZ §14).

Endpoints:
  GET  /                         -> web UI
  GET  /health                   -> liveness + which LLM provider is active
  GET  /api/cases                -> available synthetic demo cases
  POST /api/analyze              -> run pipeline (synthetic case or file upload)
  GET  /api/analysis/{id}/report.pdf
  GET  /api/analysis/{id}/waveform.svg
  GET  /api/analysis/{id}/graph.png
  GET  /api/analysis/{id}/conclusion?lang=ru|uz

Everything under /api/analyze and /api/analysis needs an API key (``X-API-Key``, the same
keys as /scans in ``$ECGSCAN_API_KEYS``); a key sees only its own analyses (D98). The web
UI, /health and the demo case list stay open.

Core layers (1-5) run locally; the Layer-6 narrative comes from a local
OpenAI-compatible model, from Claude, or from the deterministic stub (TZ §12).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from starlette.background import BackgroundTask

from cardiolens.api import service
from cardiolens.api.guard import RequestGuard
from cardiolens.calibration import TemperatureScaler, expected_calibration_error
from cardiolens.config import load_app_config
from cardiolens.feedback import DEFAULT_FEEDBACK_PATH, Feedback, FeedbackStore
from cardiolens.layer6_interpret import provider_status
from cardiolens.outputs import (
    format_conclusion,
    render_causal_graph,
    render_waveform,
    result_to_dict,
    write_pdf,
)
from ecgscan_service.router import actor
from ecgscan_service.router import router as scans_router

app = FastAPI(title="CardioLens", version="0.1.0")
# Key check and body cap before any multipart parsing (D98); uploads are 64 MB at most.
app.add_middleware(RequestGuard, max_body_bytes=service.MAX_UPLOAD_BYTES + 1024 * 1024)
# ECG scanner (photo/scan input, TZ scanner §4, SD-004): /scans, own API-key auth.
app.include_router(scans_router)

_STATIC = Path(__file__).parent / "static"


def _cleanup(tmp_dir: str) -> BackgroundTask:
    """Delete a temp dir after the FileResponse has been sent (no disk leak)."""
    return BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)


def _feedback_store() -> FeedbackStore:
    return FeedbackStore.at(os.environ.get("CARDIOLENS_FEEDBACK_PATH", DEFAULT_FEEDBACK_PATH))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Liveness plus the Layer-6 narrative source the next analysis will use (TZ §12)."""
    return {
        "status": "ok", "service": "cardiolens", "version": "0.1.0",
        "llm": provider_status(load_app_config().llm),
    }


@app.get("/api/cases")
def cases() -> dict:
    return {"cases": sorted(service.DEMO_CASES)}


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
    files: list[UploadFile] = File(default=[]),
    who: str = Depends(actor),
) -> JSONResponse:
    try:
        if files:
            uploads, budget = [], service.MAX_UPLOAD_BYTES
            for f in files:
                # A running byte budget: never more than the limit in memory (D98).
                content = await f.read(budget + 1)
                budget -= len(content)
                if budget < 0:
                    raise ValueError(f"upload too large (> {service.MAX_UPLOAD_BYTES} bytes)")
                uploads.append(service.UploadFile(filename=f.filename or "upload", content=content))
            sig = service.signal_from_uploads(
                uploads, sampling_rate_hz=sampling_rate_hz, unit=unit
            )
        elif case:
            sig = service.signal_from_case(case)
        else:
            raise ValueError("provide a 'case' or upload 'files'")

        service.apply_patient_overrides(
            sig, age_years=age_years, age_days=age_days, sex=sex,
            potassium=potassium, calcium=calcium,
        )
        analysis_id, result = service.run_and_store(sig, mode=mode, prior_id=prior_ecg_id, owner=who)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result_to_dict(result)
    payload["analysis_id"] = analysis_id
    payload["conclusion_ru"] = format_conclusion(result, "ru")
    payload["conclusion_uz"] = format_conclusion(result, "uz")
    return JSONResponse(payload)


def _require(analysis_id: str, who: str):
    result = service.get_analysis(analysis_id, owner=who)
    if result is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return result


@app.get("/api/analysis/{analysis_id}/report.pdf")
def report_pdf(analysis_id: str, who: str = Depends(actor)) -> FileResponse:
    result = _require(analysis_id, who)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.pdf")
    write_pdf(result, path)
    return FileResponse(path, media_type="application/pdf",
                        filename=f"cardiolens_{analysis_id}.pdf",
                        background=_cleanup(tmp))


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
