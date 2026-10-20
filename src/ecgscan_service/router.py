"""``/scans`` endpoints (TZ §4, SD-004).

All endpoints need an API key (``X-API-Key``; keys in ``$ECGSCAN_API_KEYS`` as
``name:key,name2:key2``; the same keys guard CardioLens ``/api``). The key's name is the
actor written to the audit log, and a key sees only the scans it submitted unless it is
listed in ``access.admins`` (D98). With no keys configured the API refuses every request
— it never runs open.
"""

from __future__ import annotations

import hmac
import os
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ecgscan.i18n import LANGS, t
from ecgscan_service.service import ScanError, ScanService
from ecgscan_service.store import ScanRow

KEYS_ENV = "ECGSCAN_API_KEYS"

router = APIRouter(prefix="/scans", tags=["scanner"])


@lru_cache(maxsize=1)
def get_service() -> ScanService:
    return ScanService()


def _keys() -> dict[str, str]:
    out = {}
    for item in os.environ.get(KEYS_ENV, "").split(","):
        name, _, key = item.strip().partition(":")
        if name and key:
            out[key] = name
    return out


def key_name(x_api_key: str | None) -> str | int:
    """The key's name, or the HTTP status refusing it (503: no keys configured, 401)."""
    keys = _keys()
    if not keys:
        return 503
    found = None
    for key, name in keys.items():      # every key compared: no early exit on a match
        if x_api_key is not None and hmac.compare_digest(x_api_key.encode(), key.encode()):
            found = name
    return found if found is not None else 401


def actor(x_api_key: str | None = Header(default=None)) -> str:
    who = key_name(x_api_key)
    if who == 503:
        raise HTTPException(503, "API keys are not configured")
    if isinstance(who, int):
        raise HTTPException(401, "invalid or missing X-API-Key")
    return who


def _http(exc: ScanError) -> HTTPException:
    return HTTPException(exc.code, str(exc))


def messages(row: ScanRow) -> dict:
    """Human-readable status, reasons and warnings in RU and UZ."""
    res = row.result or {}
    out = {}
    for lang in LANGS:
        status = res.get("status", row.status) if row.status not in ("queued", "processing", "failed") else row.status
        out[lang] = {
            "status": t(f"status.{status}", lang),
            "reasons": [t(f"reject.{c}", lang) for c in res.get("reject_reasons", [])],
            "warnings": [t(f"warning.{c}", lang) for c in res.get("warnings", [])],
        }
    return out


def view(row: ScanRow) -> dict:
    return {"scan_id": row.id, "status": row.status, "created_at": row.created_at, "updated_at": row.updated_at,
            "error": row.error, "analysis_id": row.analysis_id, "messages": messages(row), "result": row.result}


@router.post("", status_code=202)
async def submit_scan(file: UploadFile = File(...), paper_speed_mm_s: float | None = Form(None),
                      gain_mm_mv: float | None = Form(None), layout: str | None = Form(None),
                      client: Literal["flutter", "telegram_bot", "api", "web", "cli"] | None = Form(None),
                      device: str | None = Form(None),
                      header_masked: bool = Form(False), who: str = Depends(actor),
                      svc: ScanService = Depends(get_service)) -> dict:
    data = await file.read(svc.settings.max_bytes + 1)
    hints = {"paper_speed_mm_s": paper_speed_mm_s, "gain_mm_mv": gain_mm_mv, "layout": layout,
             "client": client, "device": device, "header_masked": header_masked}
    try:
        scan_id = svc.submit(data, owner=who, hints=hints)
    except ScanError as exc:
        raise _http(exc) from exc
    return {"scan_id": scan_id, "status": "queued"}


@router.get("/config/capture-qc")
def capture_qc_config(who: str = Depends(actor)) -> dict:
    """Frame-check thresholds and hint texts for the camera screen (TZ §6.1, SD-014)."""
    from ecgscan.capture_qc import HINTS, config
    from ecgscan.i18n import strings

    return {"config": config(), "hints": {lang: {h: strings(lang)[f"qc.{h}"] for h in (*HINTS, "ready")}
                                          for lang in LANGS}}


@router.get("/{scan_id}")
def get_scan(scan_id: str, who: str = Depends(actor), svc: ScanService = Depends(get_service)) -> dict:
    try:
        return view(svc.get(scan_id, who))
    except ScanError as exc:
        raise _http(exc) from exc


class ConfirmBody(BaseModel):
    decision: Literal["accepted", "corrected", "retake"]
    paper_speed_mm_s: float | None = None
    gain_mm_mv: float | None = None
    layout: str | None = None
    comment: str | None = None


@router.post("/{scan_id}/confirm")
def confirm_scan(scan_id: str, body: ConfirmBody, who: str = Depends(actor),
                 svc: ScanService = Depends(get_service)) -> dict:
    try:
        return view(svc.confirm(scan_id, actor=who, **body.model_dump()))
    except ScanError as exc:
        raise _http(exc) from exc


class AnalyzeBody(BaseModel):
    mode: Literal["A", "B"] | None = None


@router.post("/{scan_id}/analysis")
def analyze_scan(scan_id: str, body: AnalyzeBody | None = None, who: str = Depends(actor),
                 svc: ScanService = Depends(get_service)) -> dict:
    try:
        return svc.analyze(scan_id, actor=who, mode=body.mode if body else None)
    except ScanError as exc:
        raise _http(exc) from exc


@router.get("/{scan_id}/audit")
def scan_audit(scan_id: str, who: str = Depends(actor), svc: ScanService = Depends(get_service)) -> list[dict]:
    try:
        svc.get(scan_id, who)
    except ScanError as exc:
        raise _http(exc) from exc
    return svc.store.audit_log(scan_id)


@router.get("/{scan_id}/artifacts/{name}")
def scan_artifact(scan_id: str, name: str, who: str = Depends(actor),
                  svc: ScanService = Depends(get_service)) -> Response:
    try:
        data, media = svc.artifact(scan_id, name, who)
    except ScanError as exc:
        raise _http(exc) from exc
    return Response(data, media_type=media, headers={"Cache-Control": "no-store"})
