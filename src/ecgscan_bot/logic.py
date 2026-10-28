"""Bot behaviour without Telegram: whitelist, intake rules, API calls, the RU/UZ verdict.

Two modes (``bot.yaml`` ``mode`` or ``ECGSCAN_BOT_MODE``): ``analyze`` (D161, the default)
answers with the CardioVision conclusion and the PDF through ``/api/analyze`` — the same
route as the web page, the scanner first and then the CardioLens readers; ``scan``
(SD-041) answers with the scanner's own verdict through ``/scans``.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from ecgscan.config import load_yaml
from ecgscan.i18n import t

ALLOWED_ENV = "ECGSCAN_BOT_ALLOWED"
API_KEY_ENV = "ECGSCAN_BOT_API_KEY"
API_URL_ENV = "ECGSCAN_BOT_API_URL"
MODE_ENV = "ECGSCAN_BOT_MODE"
IMAGE_MIME = ("image/jpeg", "image/png", "image/tiff")                     # what /scans takes
ANALYZE_MIME = IMAGE_MIME + ("image/heic", "image/heif", "application/pdf")  # /api/analyze reads these too (D158)


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
    mode: str = "analyze"

    @classmethod
    def load(cls) -> BotConfig:
        c = load_yaml("bot")
        allowed = {int(x) for x in os.environ.get(ALLOWED_ENV, "").replace(" ", "").split(",") if x}
        return cls(api_url=os.environ.get(API_URL_ENV, c["api_url"]).rstrip("/"), max_mb=float(c["max_mb"]),
                   poll_interval_s=float(c["poll_interval_s"]), poll_timeout_s=float(c["poll_timeout_s"]),
                   send_overlay=bool(c["send_overlay"]), public_url=load_yaml("service").get("public_url"),
                   allowed=allowed, api_key=os.environ.get(API_KEY_ENV),
                   mode=os.environ.get(MODE_ENV, c.get("mode", "analyze")))


# The photo's caption names the patient: "54 м", "ж 7", "erkak 63" (D161).
_SEX_WORDS = {"male": ("м", "муж", "мужской", "мужчина", "male", "m", "erkak", "e"),
              "female": ("ж", "жен", "женский", "женщина", "female", "f", "ayol", "a")}


_MONTH_WORDS = ("мес", "месяц", "месяца", "месяцев", "oy", "oylik", "month", "months", "mo")
_DAY_WORDS = ("дн", "дней", "дня", "день", "kun", "kunlik", "day", "days", "d")


@dataclass(frozen=True)
class Patient:
    age_years: float | None = None
    age_days: float | None = None      # infants: "3 мес", "10 дней" (the norms differ by days)
    sex: str | None = None

    @property
    def given(self) -> bool:
        return (self.age_years is not None or self.age_days is not None) and self.sex is not None


SPEEDS_MM_S = (12.5, 25.0, 50.0)
GAINS_MM_MV = (5.0, 10.0, 20.0)
_SPEED_RE = re.compile(r"(\d{1,2}(?:[.,]5)?)\s*(?:мм|mm)\s*/\s*(?:сек|с|sec|s)(?![\w])", re.IGNORECASE)
_GAIN_RE = re.compile(r"(\d{1,2})\s*(?:мм|mm)\s*/\s*(?:мв|mv)(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class PaperScale:
    """The paper speed and gain the doctor gave (the caption or a button); None: read from
    the photo. Only the standard values are taken."""

    paper_speed_mm_s: float | None = None
    gain_mm_mv: float | None = None


def parse_scale(text: str | None) -> PaperScale:
    """"50 мм/с", "20 мм/мВ", "25 mm/s 10 mm/mV" in a caption; a number needs its unit."""
    if not text:
        return PaperScale()
    low = text.lower()
    speeds = [float(v.replace(",", ".")) for v in _SPEED_RE.findall(low)]
    gains = [float(v) for v in _GAIN_RE.findall(low)]
    return PaperScale(next((v for v in speeds if v in SPEEDS_MM_S), None), next((v for v in gains if v in GAINS_MM_MV), None))


def parse_caption(text: str | None) -> Patient:
    """The patient from a caption: the first number 0–120 is the age in years, unless a
    month/day word follows it ("3 мес", "10 kun"); the first sex word is the sex. Anything
    else is ignored, anything missing stays None. Speed and gain ("50 мм/с") are not ages."""
    if not text:
        return Patient()
    low = _GAIN_RE.sub(" ", _SPEED_RE.sub(" ", text.lower())).replace(",", ".")
    age_years = age_days = None
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([^\W\d_]*)", low):
        value, unit = float(m.group(1)), m.group(2)
        if unit in _MONTH_WORDS and age_days is None:
            age_days = value * 30.4
        elif unit in _DAY_WORDS and age_days is None:
            age_days = value
        elif age_years is None and age_days is None and 0 <= value <= 120:
            age_years = value
    words = re.findall(r"[^\W\d_]+", low)
    sex = next((s for w in words for s, forms in _SEX_WORDS.items() if w in forms), None)
    return Patient(age_years=None if age_days is not None else age_years, age_days=age_days, sex=sex)


def lang_of(language_code: str | None) -> str:
    """Uzbek for Telegram clients set to Uzbek, Russian otherwise."""
    return "uz" if (language_code or "").lower().startswith("uz") else "ru"


def allowed(cfg: BotConfig, user_id: int | None, chat_type: str | None = "private") -> bool:
    """A whitelisted user in a private chat: in a group the verdict (and the overlay)
    would reach every member (D136)."""
    return user_id is not None and user_id in cfg.allowed and chat_type == "private"


def intake(cfg: BotConfig, *, kind: str, mime: str | None, size: int | None, lang: str) -> str | None:
    """What to answer *instead of* scanning: ``kind`` is 'photo' (compressed by Telegram),
    'document' or 'other'. ``None`` means: download the file and scan it."""
    if kind == "photo":
        return t("bot.send_as_document", lang)       # Telegram's photo is ~4 px/mm, below 8 (SD-009, D72)
    if kind != "document" or mime not in (ANALYZE_MIME if cfg.mode == "analyze" else IMAGE_MIME):
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


class AnalyzeRefused(Exception):
    """The analysis said no in words (HTTP 400): the text is the reply."""


class AnalyzeApi:
    """The CardioVision calls the bot makes in ``analyze`` mode (D161): ``/api/analyze`` with
    ``client=telegram`` (the refusals keep their Telegram wording) and the PDF report."""

    def __init__(self, cfg: BotConfig, http=None):
        import httpx

        self.cfg = cfg
        self.http = http or httpx.Client(base_url=cfg.api_url, timeout=180.0)
        self.headers = {"X-API-Key": cfg.api_key or ""}

    def analyze(self, files: list[tuple[str, bytes]], *, patient: Patient = Patient(),
                scale: PaperScale = PaperScale()) -> dict:
        """One analysis of one or several files (an album is one record: pages of a device,
        parts of a strip, parts of a page on the capture board — D155)."""
        form = {"mode": "A", "client": "telegram"}
        if scale.paper_speed_mm_s is not None:
            form["paper_speed_mm_s"] = f"{scale.paper_speed_mm_s:g}"
        if scale.gain_mm_mv is not None:
            form["gain_mm_mv"] = f"{scale.gain_mm_mv:g}"
        if patient.age_years is not None:
            form["age_years"] = str(patient.age_years)
        if patient.age_days is not None:
            form["age_days"] = str(patient.age_days)
        if patient.sex:
            form["sex"] = patient.sex
        r = self.http.post("/api/analyze", files=[("files", (name, data)) for name, data in files],
                           data=form, headers=self.headers)
        if r.status_code == 400:
            raise AnalyzeRefused(r.json().get("detail", r.text))
        r.raise_for_status()
        return r.json()

    def pdf(self, analysis_id: str) -> bytes | None:
        r = self.http.get(f"/api/analysis/{analysis_id}/report.pdf", headers=self.headers)
        return r.content if r.status_code == 200 else None

    def overlay(self, analysis_id: str) -> bytes | None:
        """The scanner's confirmation overlay of the photo (D163): the traces it read, drawn
        over the page. None for a photo the CardioLens readers read, or a signal file."""
        r = self.http.get(f"/api/analysis/{analysis_id}/overlay.jpg", headers=self.headers)
        return r.content if r.status_code == 200 else None


def scale_line(body: dict, scale: PaperScale, lang: str) -> str | None:
    """What the reply says about the time and amplitude scale: the values the doctor gave,
    or, when the speed was neither read nor given, that 25 mm/s was assumed (D180)."""
    parts = []
    if scale.paper_speed_mm_s is not None:
        parts.append(t("bot.scale_speed", lang, speed=f"{scale.paper_speed_mm_s:g}"))
    if scale.gain_mm_mv is not None:
        parts.append(t("bot.scale_gain", lang, gain=f"{scale.gain_mm_mv:g}"))
    if parts:
        return t("bot.scale_given", lang, values=", ".join(parts))
    if body.get("speed_note_ru"):
        return t("bot.speed_assumed", lang)
    return None


def scale_buttons(scale: PaperScale, lang: str = "ru") -> list[list[tuple[str, str]]]:
    """Rows of (label, choice) under a reply: re-run with another speed or gain. The value in
    use is marked; "auto" returns the choice to what the photo shows."""
    auto, speed_unit, gain_unit = ("avto", "mm/s", "mm/mV") if lang == "uz" else ("авто", "мм/с", "мм/мВ")

    def row(values, current, unit, key):
        out = [(("✓ " if current is None else "") + auto, f"{key}=")]
        out += [(("✓ " if current == v else "") + f"{v:g} {unit}", f"{key}={v:g}") for v in values]
        return out

    return [row((25.0, 50.0), scale.paper_speed_mm_s, speed_unit, "s"), row((10.0, 20.0), scale.gain_mm_mv, gain_unit, "g")]


def apply_choice(scale: PaperScale, choice: str) -> PaperScale:
    """A button's choice ("s=50", "g=", ...) applied to the scale in use."""
    key, _, value = choice.partition("=")
    v = float(value) if value else None
    if key == "s" and (v is None or v in SPEEDS_MM_S):
        return PaperScale(v, scale.gain_mm_mv)
    if key == "g" and (v is None or v in GAINS_MM_MV):
        return PaperScale(scale.paper_speed_mm_s, v)
    return scale


def conclusion_reply(body: dict, lang: str, *, patient_given: bool = True) -> str:
    """The reply in ``analyze`` mode: the critical alerts first, then the conclusion as the
    web page prints it (it ends with the disclaimer), why a doctor must look, if so, and a
    reminder when the caption named no age or sex (the norms then are an adult's)."""
    lines = []
    alerts = body.get("critical_alerts") or []
    if alerts:
        lines.append(t("bot.alerts", lang))
        lines += [f"⚠ {a['label_' + lang]}" for a in alerts]
        lines.append("")
    lines.append(body["conclusion_uz" if lang == "uz" else "conclusion_ru"])
    if body.get("mode") == "escalate" and body.get("escalation_reason"):
        lines += ["", t("bot.escalate", lang, reason=body["escalation_reason"])]
    if not patient_given:
        lines += ["", t("bot.no_patient", lang)]
    return "\n".join(lines)


def handle_analysis(cfg: BotConfig, api: AnalyzeApi, files: list[tuple[str, bytes]], lang: str,
                    caption: str | None = None, scale: PaperScale | None = None) -> tuple[str, str | None]:
    """Analyse through ``/api/analyze``: (reply text, analysis id). A refusal is said in the
    readers' words; server trouble is answered, never raised. ``scale``: speed and gain
    chosen by a button; otherwise those written in the caption, if any."""
    patient = parse_caption(caption)
    scale = scale if scale is not None else parse_scale(caption)
    try:
        body = api.analyze(files, patient=patient, scale=scale)
    except AnalyzeRefused as e:
        return str(e), None
    except Exception:  # network, HTTP error: the user gets a plain answer
        return t("bot.server_error", lang), None
    text = conclusion_reply(body, lang, patient_given=patient.given)
    line = scale_line(body, scale, lang)
    return (f"{text}\n\n{line}" if line else text), body.get("analysis_id")


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
