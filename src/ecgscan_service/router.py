"""``/scans`` endpoints (TZ §4, SD-004).

All endpoints need an API key (``X-API-Key``; keys in ``$ECGSCAN_API_KEYS`` as
``name:key,name2:key2``; the same keys guard CardioLens ``/api``). The key's name is the
actor written to the audit log, and a key sees only the scans it submitted unless it is
listed in ``access.admins`` (D136). With no keys configured the API refuses every request
— it never runs open.
"""

from __future__ import annotations

import hmac
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from ecgscan.config import load_yaml
from ecgscan.i18n import LANGS, strings, t
from ecgscan.layout.devices import device_profiles
from ecgscan_service.miniapp import (
    TICKET_HEADER,
    Inbox,
    InboxFull,
    Ticket,
    image_kind,
    verify_ticket,
)
from ecgscan_service.service import ScanError, ScanService
from ecgscan_service.store import ScanRow

KEYS_ENV = "ECGSCAN_API_KEYS"

router = APIRouter(prefix="/scans", tags=["scanner"])
# SD-136: pages Telegram opens as a Mini App. Telegram has no clinic key, so these are open; they are
# static, read nothing from the server and (by the headers below) cannot reach any network.
public_router = APIRouter(prefix="/miniapp", tags=["miniapp"])
_STATIC = Path(__file__).parent / "static"
# Inline script and style only, the page's own camera, nothing else: no connection, no form, no outside file.
PROBE_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "img-src blob:; media-src blob: mediastream:; connect-src 'none'; "
                                "form-action 'none'; base-uri 'none'"),
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


# The capture page (SD-137) may also post to its own two addresses, and nowhere else.
CAPTURE_HEADERS = {**PROBE_HEADERS,
                   "Content-Security-Policy": PROBE_HEADERS["Content-Security-Policy"].replace(
                       "connect-src 'none'", "connect-src 'self'")}
_PAGE_TEXTS = ("title", "lead", "tips", "shoot", "pick", "uploading", "checking", "size", "hints_title", "retake",
               "send_anyway", "sent", "no_ticket", "expired", "too_big", "bad_format", "busy", "failed", "retry")


@lru_cache(maxsize=1)
def get_inbox() -> Inbox:
    return Inbox(int(load_yaml("service")["miniapp"]["max_waiting"]))


def _lang(lang: str | None) -> str:
    return lang if lang in LANGS else "ru"


def _ticket(request: Request) -> Ticket:
    """The live, unused ticket of the request, from its header alone (the body is not touched)."""
    tk = verify_ticket(request.headers.get(TICKET_HEADER), _keys(),
                       max_minutes=float(load_yaml("service")["miniapp"]["max_ticket_minutes"]))
    if tk is None or get_inbox().used(tk.id):
        raise HTTPException(401, "invalid, expired or used ticket")
    return tk


def photo_hints(data: bytes, lang: str) -> dict:
    """What the frame check says about an uploaded photo, in words (SD-137); nothing is kept."""
    import cv2
    import numpy as np

    from ecgscan.capture_qc import check_frame

    c = load_yaml("service")["miniapp"]
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:                       # a format OpenCV does not read: the analysis decides
        return {"checked": False, "width": None, "height": None, "px_per_mm": None, "hints": []}
    h, w = bgr.shape[:2]
    k = min(1.0, int(c["check_width"]) / w)
    small = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else bgr
    qc = check_frame(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale_to_capture=w / small.shape[1])
    s = strings(lang)
    hints = [s[f"qc.frame_{qc.frame_advice}"] if code == "frame" and qc.frame_advice else s[f"qc.{code}"]
             for code in qc.hints]
    if qc.px_per_mm is None and max(w, h) < int(c["small_side_px"]):
        hints.append(s["miniapp.small"].format(px=f"{max(w, h) / 297:.1f}"))
    return {"checked": True, "width": w, "height": h, "px_per_mm": qc.px_per_mm, "hints": hints}


@public_router.get("/capture", response_class=HTMLResponse)
def miniapp_capture(lang: str = "ru") -> HTMLResponse:
    """The capture page: the texts of the language and the upload limit are put into the static page."""
    import json

    lang = _lang(lang)
    s = strings(lang)
    page = {"lang": lang, "max_mb": load_yaml("service")["upload"]["max_mb"],
            "texts": {f"miniapp.{k}": s[f"miniapp.{k}"] for k in _PAGE_TEXTS}}
    blob = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    html = (_STATIC / "miniapp_capture.html").read_text(encoding="utf-8").replace("/*__PAGE__*/{}", blob, 1)
    return HTMLResponse(html, headers=CAPTURE_HEADERS)


@public_router.post("/upload")
async def miniapp_upload(request: Request, lang: str = "ru") -> dict:
    """The photo as the phone made it (the raw request body). The ticket, its count of uploads and the
    declared size are checked before a byte of the body is read; the body is read as a stream, cut off
    at the upload limit and dropped at its first bytes if they are not an image the scanner reads."""
    import asyncio

    tk = _ticket(request)
    conf = load_yaml("service")
    cap = int(conf["upload"]["max_mb"] * 1024 * 1024)
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        raise HTTPException(413, "file too large")
    if not get_inbox().attempt(tk, int(conf["miniapp"]["max_uploads_per_ticket"])):
        raise HTTPException(401, "invalid, expired or used ticket")
    chunks, size, known = [], 0, False
    async for chunk in request.stream():
        size += len(chunk)
        if size > cap:
            raise HTTPException(413, "file too large")
        chunks.append(chunk)
        if not known and size >= 8:
            if f"image/{image_kind(b''.join(chunks)[:8])}" not in conf["upload"]["mime"]:
                raise HTTPException(415, "not a photo the scanner reads")
            known = True
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty file")
    if not known:
        raise HTTPException(415, "not a photo the scanner reads")
    info = await asyncio.get_running_loop().run_in_executor(None, photo_hints, data, _lang(lang))
    try:
        get_inbox().put(tk, data)
    except InboxFull:
        raise HTTPException(429, "too many photos are waiting") from None
    except KeyError:
        raise HTTPException(401, "invalid, expired or used ticket") from None
    return info


@public_router.post("/send")
def miniapp_send(request: Request) -> dict:
    """The doctor is done with the photo of this ticket: the bot may take it."""
    if not get_inbox().send(_ticket(request)):
        raise HTTPException(409, "nothing uploaded for this ticket")
    return {"sent": True}


@public_router.get("/probe", response_class=HTMLResponse)
def miniapp_probe() -> HTMLResponse:
    """The phone probe (SD-134): what a page inside Telegram gets from the camera, as numbers on the screen."""
    return HTMLResponse((_STATIC / "miniapp_probe.html").read_text(encoding="utf-8"), headers=PROBE_HEADERS)


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
            "warnings": [t(f"warning.{c}", lang, **_fmt(res, lang)) for c in res.get("warnings", [])],
        }
    return out


def _fmt(res: dict, lang: str) -> dict:
    """Placeholders of the warning texts: the photo's px/mm (LOW_RESOLUTION_COARSE, D170)."""
    ppm = (res.get("quality") or {}).get("px_per_mm")
    if ppm is None:
        return {"ppm": "?"}
    s = f"{ppm:.1f}"
    return {"ppm": s.replace(".", ",") if lang == "ru" else s}


def view(row: ScanRow) -> dict:
    return {"scan_id": row.id, "status": row.status, "created_at": row.created_at, "updated_at": row.updated_at,
            "error": row.error, "analysis_id": row.analysis_id, "messages": messages(row), "result": row.result}


def _capture_form(raw: str | None) -> dict | None:
    """``capture``: ecgscan.contract.Capture as JSON (TZ addendum §18.13, SD-071); a malformed
    one is refused, not dropped: the record would silently lose the phone's checks."""
    if not raw:
        return None
    from pydantic import ValidationError

    from ecgscan.contract import Capture

    try:
        return Capture.model_validate_json(raw).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise HTTPException(422, "capture: not a valid capture record") from exc


def _check_layout(layout: str | None) -> None:
    """A layout hint or correction must name a registry layout or its alias (S2)."""
    from ecgscan.layout.registry import layout_name

    if layout is None:
        return
    try:
        layout_name(layout)
    except KeyError as exc:
        raise HTTPException(422, f"layout: unknown {layout!r}") from exc


@router.post("", status_code=202)
async def submit_scan(file: UploadFile = File(...), parts: list[UploadFile] | None = File(None),
                      paper_speed_mm_s: float | None = Form(None),
                      gain_mm_mv: float | None = Form(None), layout: str | None = Form(None),
                      client: Literal["flutter", "telegram_bot", "api", "web", "cli"] | None = Form(None),
                      device: str | None = Form(None),
                      header_masked: bool = Form(False), capture: str | None = Form(None),
                      device_profile: str | None = Form(None), pages: bool = Form(False),
                      who: str = Depends(actor), svc: ScanService = Depends(get_service)) -> dict:
    # ``parts``: more photos after ``file`` — the same page in parts on a capture board (SD-066),
    # or, with ``pages``, the next sheets of a study on several pages (S2, TZ 18.4).
    _check_layout(layout)
    if device_profile is not None and device_profile not in device_profiles():
        raise HTTPException(422, f"device_profile: unknown {device_profile!r}")
    if parts and len(parts) + 1 > svc.settings.max_parts:
        raise HTTPException(400, f"1 to {svc.settings.max_parts} photos per scan")
    data: bytes | list[bytes] = await file.read(svc.settings.max_bytes + 1)
    if parts:
        data = [data] + [await f.read(svc.settings.max_bytes + 1) for f in parts]
    hints = {"paper_speed_mm_s": paper_speed_mm_s, "gain_mm_mv": gain_mm_mv, "layout": layout,
             "client": client, "device": device, "header_masked": header_masked,
             "capture": _capture_form(capture), "device_profile": device_profile, "pages": pages}
    try:
        scan_id = svc.submit(data, owner=who, hints=hints)
    except ScanError as exc:
        raise _http(exc) from exc
    return {"scan_id": scan_id, "status": "queued"}


@router.get("/config/devices")
def devices_config(who: str = Depends(actor)) -> dict:
    """The recorder profiles the app picks from: name, layouts, personal-data zones (S3)."""
    from ecgscan.layout.devices import device_profiles

    return {"devices": [p.public() for p in device_profiles().values()]}


@router.get("/config/features")
def features_config(who: str = Depends(actor)) -> dict:
    """Which P2 features are on (TZ 18.11, SD-097), so the app shows only those: today none."""
    from ecgscan_service.extensions import features

    return {"features": features()}


@router.get("/config/capture-qc")
def capture_qc_config(who: str = Depends(actor)) -> dict:
    """Frame-check thresholds and hint texts for the camera screen (TZ §6.1, SD-014)."""
    from ecgscan.capture_qc import FRAME_ADVICE, HINTS, config
    from ecgscan.i18n import strings

    return {"config": config(), "hints": {lang: {h: strings(lang)[f"qc.{h}"] for h in (*HINTS, "ready")}
                                          for lang in LANGS},
            # SD-133: a more exact text for the "frame" hint, by FrameQC.frame_advice
            "frame_advice": {lang: {a: strings(lang)[f"qc.frame_{a}"] for a in FRAME_ADVICE} for lang in LANGS}}


@router.get("/miniapp/inbox/{ticket_id}")
def miniapp_take(ticket_id: str, who: str = Depends(actor)) -> Response:
    """The bot takes the photo of a ticket it signed, once (SD-137): 200 the bytes (the server forgets
    them), 204 nothing ready yet, 404 already taken."""
    got = get_inbox().take(ticket_id, who)
    if got is False:
        raise HTTPException(404, "already taken")
    if got is None:
        return Response(status_code=204)
    return Response(got, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


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
    _check_layout(body.layout)
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


@router.get("/{scan_id}/export")
def scan_export(scan_id: str, format: str = "json", who: str = Depends(actor),
                svc: ScanService = Depends(get_service)) -> Response:
    """The read signal (TZ 18.10): ``json`` (the CardioVision contract), ``csv`` or ``wfdb`` (zip)."""
    try:
        data, media, name = svc.export(scan_id, format, who)
    except ScanError as exc:
        raise _http(exc) from exc
    return Response(data, media_type=media, headers={"Cache-Control": "no-store",
                                                     "Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/{scan_id}/artifacts/{name}")
def scan_artifact(scan_id: str, name: str, who: str = Depends(actor),
                  svc: ScanService = Depends(get_service)) -> Response:
    try:
        data, media = svc.artifact(scan_id, name, who)
    except ScanError as exc:
        raise _http(exc) from exc
    return Response(data, media_type=media, headers={"Cache-Control": "no-store"})
