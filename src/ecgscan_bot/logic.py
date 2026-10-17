"""Bot behaviour without Telegram: whitelist, intake rules, API calls, the RU/UZ verdict."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ecgscan.config import load_yaml
from ecgscan.i18n import t

ALLOWED_ENV = "ECGSCAN_BOT_ALLOWED"
API_KEY_ENV = "ECGSCAN_BOT_API_KEY"
IMAGE_MIME = ("image/jpeg", "image/png", "image/tiff")


@dataclass
class BotConfig:
    api_url: str
    max_mb: float
    poll_interval_s: float
    poll_timeout_s: float
    send_overlay: bool
    public_url: str | None
    allowed: set[int] = field(default_factory=set)
    api_key: str | None = None

    @classmethod
    def load(cls) -> BotConfig:
        c = load_yaml("bot")
        allowed = {int(x) for x in os.environ.get(ALLOWED_ENV, "").replace(" ", "").split(",") if x}
        return cls(api_url=c["api_url"].rstrip("/"), max_mb=float(c["max_mb"]),
                   poll_interval_s=float(c["poll_interval_s"]), poll_timeout_s=float(c["poll_timeout_s"]),
                   send_overlay=bool(c["send_overlay"]), public_url=load_yaml("service").get("public_url"),
                   allowed=allowed, api_key=os.environ.get(API_KEY_ENV))


def lang_of(language_code: str | None) -> str:
    """Uzbek for Telegram clients set to Uzbek, Russian otherwise."""
    return "uz" if (language_code or "").lower().startswith("uz") else "ru"


def allowed(cfg: BotConfig, user_id: int | None) -> bool:
    return user_id is not None and user_id in cfg.allowed


def intake(cfg: BotConfig, *, kind: str, mime: str | None, size: int | None, lang: str) -> str | None:
    """What to answer *instead of* scanning: ``kind`` is 'photo' (compressed by Telegram),
    'document' or 'other'. ``None`` means: download the file and scan it."""
    if kind == "photo":
        return t("bot.send_as_document", lang)       # Telegram's photo is ~4 px/mm, below 8 (SD-009, D72)
    if kind != "document" or mime not in IMAGE_MIME:
        return t("bot.not_an_image", lang)
    if size is not None and size > cfg.max_mb * 1024 * 1024:
        return t("bot.too_large", lang, max_mb=int(cfg.max_mb))
    return None


class ScanApi:
    """The /scans calls the bot makes. ``http`` is an ``httpx.Client`` (or a FastAPI TestClient)."""

    def __init__(self, cfg: BotConfig, http=None):
        import httpx

        self.cfg = cfg
        self.http = http or httpx.Client(base_url=cfg.api_url, timeout=30.0)
        self.headers = {"X-API-Key": cfg.api_key or ""}

    def submit(self, data: bytes, filename: str) -> str:
        r = self.http.post("/scans", files={"file": (filename, data)}, data={"client": "telegram_bot"},
                           headers=self.headers)
        r.raise_for_status()
        return r.json()["scan_id"]

    def wait(self, scan_id: str) -> dict | None:
        """The scan once it is done, or None after ``poll_timeout_s``."""
        end = time.monotonic() + self.cfg.poll_timeout_s
        while time.monotonic() < end:
            r = self.http.get(f"/scans/{scan_id}", headers=self.headers)
            r.raise_for_status()
            body = r.json()
            if body["status"] not in ("queued", "processing"):
                return body
            time.sleep(self.cfg.poll_interval_s)
        return None

    def overlay(self, scan_id: str) -> bytes | None:
        r = self.http.get(f"/scans/{scan_id}/artifacts/trace_overlay_preview", headers=self.headers)
        return r.content if r.status_code == 200 else None


def link(cfg: BotConfig, scan_id: str) -> str:
    return f"{cfg.public_url.rstrip('/')}/scans/{scan_id}" if cfg.public_url else f"scan_id: {scan_id}"


def verdict(cfg: BotConfig, body: dict, lang: str) -> str:
    """The reply: status, calibration, layout, reasons or warnings, the link, the disclaimer."""
    msg = body["messages"][lang]
    lines = [msg["status"]]
    res = body.get("result") or {}
    if res.get("status") in ("ok", "needs_confirmation"):
        cal = res["calibration"]
        if cal.get("paper_speed_mm_s") and cal.get("gain_mm_mv"):
            lines.append(t("result.calibration", lang, speed=f"{cal['paper_speed_mm_s']:g}",
                           gain=f"{cal['gain_mm_mv']:g}"))
        else:
            lines.append(t("result.calibration_unknown", lang))
        if res.get("layout", {}).get("name"):
            lines.append(t("result.layout", lang, layout=res["layout"]["name"]))
        lines.append(t("result.quality", lang, quality=f"{res['quality']['overall']:.2f}"))
        lines.append(t("result.digitized_note", lang))
    if msg["reasons"]:
        lines += ["", t("bot.reasons", lang)] + [f"• {x}" for x in msg["reasons"]]
    if msg["warnings"]:
        lines += ["", t("bot.warnings", lang)] + [f"• {x}" for x in msg["warnings"]]
    if body["status"] != "rejected":
        lines += ["", t("result.link", lang, url=link(cfg, body["scan_id"]))]
    lines += ["", t("bot.disclaimer", lang)]
    return "\n".join(lines)


def handle_file(cfg: BotConfig, api: ScanApi, data: bytes, filename: str, lang: str) -> tuple[str, str | None]:
    """Submit and wait: (reply text, scan id). Server trouble is answered, never raised."""
    try:
        scan_id = api.submit(data, filename)
        body = api.wait(scan_id)
    except Exception:  # network, HTTP error: the user gets a plain answer
        return t("bot.server_error", lang), None
    if body is None:
        return t("bot.timeout", lang, url=link(cfg, scan_id)), scan_id
    return verdict(cfg, body, lang), scan_id
