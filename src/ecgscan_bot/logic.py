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
PUBLIC_URL_ENV = "ECGSCAN_BOT_PUBLIC_URL"   # the site, for links sent to colleagues (D208)
GROUPS_ENV = "ECGSCAN_BOT_GROUPS"           # SD-107: chat ids of the groups the bot reads (also switched on by a button)
GROUP_RECIPIENTS_ENV = "ECGSCAN_BOT_GROUP_RECIPIENTS"   # who gets the group's results in private; default: every administrator
GROUP_CHAT_TYPES = ("group", "supergroup")
IMAGE_MIME = ("image/jpeg", "image/png", "image/tiff")                     # what /scans takes
ANALYZE_MIME = IMAGE_MIME + ("image/heic", "image/heif", "application/pdf")  # /api/analyze reads these too (D158)


def _ids(text: str) -> set[int]:
    """Comma-separated Telegram ids (a group's id is negative)."""
    out = set()
    for x in text.replace(" ", "").split(","):
        try:
            out.add(int(x))
        except ValueError:
            continue
    return out


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
    groups: set[int] = field(default_factory=set)
    group_recipients: set[int] = field(default_factory=set)
    group_notify_refusals: bool = True
    group_max_per_hour: int = 60

    @classmethod
    def load(cls) -> BotConfig:
        c = load_yaml("bot")
        g = c.get("group") or {}
        allowed = _ids(os.environ.get(ALLOWED_ENV, ""))
        return cls(api_url=os.environ.get(API_URL_ENV, c["api_url"]).rstrip("/"), max_mb=float(c["max_mb"]),
                   poll_interval_s=float(c["poll_interval_s"]), poll_timeout_s=float(c["poll_timeout_s"]),
                   send_overlay=bool(c["send_overlay"]), public_url=load_yaml("service").get("public_url"),
                   allowed=allowed, api_key=os.environ.get(API_KEY_ENV),
                   mode=os.environ.get(MODE_ENV, c.get("mode", "analyze")),
                   groups=_ids(os.environ.get(GROUPS_ENV, "")),
                   # Results of a group go to administrators only: an id outside the whitelist is dropped.
                   group_recipients=(_ids(os.environ.get(GROUP_RECIPIENTS_ENV, "")) & allowed) or allowed,
                   group_notify_refusals=bool(g.get("notify_refusals", True)),
                   group_max_per_hour=int(g.get("max_per_hour", 60)))


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


def allowed(cfg: BotConfig, user_id: int | None, chat_type: str | None = "private", store=None) -> bool:
    """A whitelisted user (an administrator) or one who joined with a live invite code (D208), in a
    private chat: in a group the verdict (and the overlay) would reach every member (D136)."""
    if user_id is None or chat_type != "private":
        return False
    return user_id in cfg.allowed or (store is not None and store.is_member(user_id))


def is_admin(cfg: BotConfig, user_id: int | None) -> bool:
    return user_id is not None and user_id in cfg.allowed


def watched_group(cfg: BotConfig, chat_id: int | None, chat_type: str | None, store=None) -> bool:
    """SD-107: a group whose ECG photos the bot reads — named in the environment, or switched on by
    an administrator's button. The bot never writes there: the result goes to administrators in private."""
    if chat_id is None or chat_type not in GROUP_CHAT_TYPES:
        return False
    return chat_id in cfg.groups or (store is not None and store.group_on(chat_id))


def message_link(chat_id: int, message_id: int | None, username: str | None = None) -> str | None:
    """A link to the group's message, for the administrator to find the photo: public groups by
    name, private supergroups by ``t.me/c``; a basic group has no message links."""
    if message_id is None:
        return None
    if username:
        return f"https://t.me/{username}/{message_id}"
    s = str(chat_id)
    return f"https://t.me/c/{s[4:]}/{message_id}" if s.startswith("-100") else None


def group_head(title: str | None, link_: str | None, lang: str) -> str:
    from ecgscan_bot.texts import tr

    head = tr("group_head", lang, title=(title or "—")[:80])
    return f"{head}\n{link_}" if link_ else head


class HourlyLimit:
    """At most ``n`` analyses an hour per group: a flood in a chat must not queue hours of work."""

    def __init__(self, n: int):
        self.n = n
        self._at: dict[int, list[float]] = {}

    def allow(self, key: int, now: float) -> bool:
        at = [x for x in self._at.get(key, []) if now - x < 3600.0]
        ok = len(at) < self.n
        if ok:
            at.append(now)
        self._at[key] = at
        return ok


# Symptoms in a caption (D208 -> D195 codes): words a doctor or a patient writes.
_SYMPTOM_WORDS = {
    "chest_pain": ("боль в груди", "боли в груди", "загрудинн", "давит в груди", "ko'krakda og'riq",
                   "ko‘krakda og‘riq", "chest pain"),
    "dyspnea": ("одышк", "задыха", "hansira", "nafas qis", "dyspn", "short of breath"),
    "syncope": ("обморок", "потеря сознания", "потерял сознание", "hushdan", "syncope", "faint"),
    "palpitations": ("сердцебиени", "перебои", "yurak urishi", "yurak tez", "palpitat"),
}
_ONSET_RE = re.compile(r"(\d{1,3})\s*(?:ч|час|часа|часов|soat|h|hours?)(?![\w])", re.IGNORECASE)


def parse_symptoms(text: str | None) -> tuple[list[str], str | None]:
    """D195 symptom codes and the onset band from a caption ("боль в груди 3 ч"); empty when none.
    The onset is read only next to a symptom: a lone number is the age (parse_caption)."""
    if not text:
        return [], None
    low = text.lower()
    found = [code for code, words in _SYMPTOM_WORDS.items() if any(w in low for w in words)]
    onset = None
    m = _ONSET_RE.search(low)
    if found and m:
        h = int(m.group(1))
        onset = "lt12h" if h < 12 else "12to48h" if h <= 48 else "gt48h"
    return found, onset


def intake(cfg: BotConfig, *, kind: str, mime: str | None, size: int | None, lang: str) -> str | None:
    """What to answer *instead of* scanning: ``kind`` is 'photo' (compressed by Telegram),
    'document' or 'other'. ``None`` means: download the file and scan it."""
    if kind == "photo":
        # D209: Telegram's photo (~1280 px, ~3-4 px/mm) is read in the coarse mode of D170 — rate and
        # rhythm, no intervals or ST, said in the reply — as the web page does. The scanner-only mode
        # still needs the original file.
        return None if cfg.mode == "analyze" else t("bot.send_as_document", lang)
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
                scale: PaperScale = PaperScale(), symptoms: list[str] | None = None,
                onset: str | None = None, note: str | None = None) -> dict:
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
        if symptoms:                               # D208 -> D195: the server keeps the known codes only
            form["symptoms"] = ",".join(symptoms)
        if onset:
            form["symptom_onset"] = onset
        if note:                                   # D209: the doctor's diagnosis / description
            form["clinical_note"] = note
        r = self.http.post("/api/analyze", files=[("files", (name, data)) for name, data in files],
                           data=form, headers=self.headers)
        if r.status_code == 400:
            raise AnalyzeRefused(r.json().get("detail", r.text))
        r.raise_for_status()
        return r.json()

    def pdf(self, analysis_id: str) -> bytes | None:
        r = self.http.get(f"/api/analysis/{analysis_id}/report.pdf", headers=self.headers)
        return r.content if r.status_code == 200 else None

    def share(self, analysis_id: str) -> str | None:
        """A signed 7-day link to the PDF without the photo (D197), absolute when the site's address
        is known (D208); None when the analysis is gone or the server says no."""
        try:
            r = self.http.post(f"/api/analysis/{analysis_id}/share", headers=self.headers)
        except Exception:
            return None
        if r.status_code != 200:
            return None
        url = r.json().get("url")
        base = os.environ.get(PUBLIC_URL_ENV) or self.cfg.public_url
        return f"{base.rstrip('/')}{url}" if base and url and url.startswith("/") else url

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
    text, aid, _ = analyse(cfg, api, files, lang, caption, scale)
    return text, aid


def analyse(cfg: BotConfig, api: AnalyzeApi, files: list[tuple[str, bytes]], lang: str,
            caption: str | None = None, scale: PaperScale | None = None,
            symptoms: list[str] | None = None, onset: str | None = None,
            note: str | None = None) -> tuple[str, str | None, dict | None]:
    """``handle_analysis`` plus the response body (to answer again in the other language, D208).
    Symptoms: those given by the buttons, else those written in the caption."""
    patient = parse_caption(caption)
    scale = scale if scale is not None else parse_scale(caption)
    if symptoms is None:
        symptoms, onset = parse_symptoms(caption)
    try:
        extra = {"symptoms": symptoms, "onset": onset} if symptoms else {}   # no symptoms: the call as before
        if note:
            extra["note"] = note
        body = api.analyze(files, patient=patient, scale=scale, **extra)
    except AnalyzeRefused as e:
        return str(e), None, None
    except Exception:  # network, HTTP error: the user gets a plain answer
        return t("bot.server_error", lang), None, None
    return render(body, lang, scale, patient.given), body.get("analysis_id"), body


def render(body: dict, lang: str, scale: PaperScale, patient_given: bool) -> str:
    text = conclusion_reply(body, lang, patient_given=patient_given)
    line = scale_line(body, scale, lang)
    return f"{text}\n\n{line}" if line else text


def symptom_buttons(selected: list[str], onset: str | None, lang: str) -> list[list[tuple[str, str]]]:
    """Rows of (label, choice): four symptom toggles, the onset, and "re-run" (D208)."""
    from ecgscan_bot.texts import ONSET_BTN, SYMPTOM_BTN, tr

    i = 1 if lang == "uz" else 0
    sym = [(("✓ " if c in selected else "") + names[i], f"y={c}") for c, names in SYMPTOM_BTN.items()]
    ons = [(("✓ " if c == onset else "") + names[i], f"o={c}") for c, names in ONSET_BTN.items()]
    return [sym[:2], sym[2:], ons, [(tr("btn_recalc_sym", lang), "go")]]


def toggle_symptom(selected: list[str], onset: str | None, choice: str) -> tuple[list[str], str | None]:
    from ecgscan_bot.texts import ONSET_BTN, SYMPTOM_BTN

    key, _, value = choice.partition("=")
    if key == "y" and value in SYMPTOM_BTN:
        selected = [c for c in selected if c != value] if value in selected else [*selected, value]
    elif key == "o" and value in ONSET_BTN:
        onset = None if onset == value else value
    return selected, onset


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


# --- D209: questions after the file -------------------------------------------------------------------

def patient_answer(text: str | None) -> Patient | None:
    """The doctor's answer to "sex and age": a Patient when it names an age or a sex, else None."""
    p = parse_caption(text)
    return p if (p.age_years is not None or p.age_days is not None or p.sex is not None) else None


def first_question(caption: str | None) -> str:
    """'patient' when the caption does not give both age and sex, else 'note'."""
    return "note" if parse_caption(caption).given else "patient"
