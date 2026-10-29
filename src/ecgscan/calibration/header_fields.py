"""What a printed ECG header says besides speed and gain (S3, TZ 18.6).

Input: the header's OCR lines (boxes grouped into lines, top to bottom). Output: the recorder's
own measurements (HR, PR, QRS, QT, QTc, axes), filters, recording date/time, recorder model,
its text conclusion, and the lines with personal data dropped.

Personal data never leaves this module: a line with a name, ID, birth date, age, sex, address
or phone label is dropped whole, and so is an unlabelled line that looks like a name (2-4
capitalised words, no digits). Only the dropped lines' count is kept. The recorder's text
conclusion is kept as the recorder's, never used as ours (TZ 18.6).

Labels cover Russian, Uzbek and English recorders and the Latin look-alikes the OCR returns for
Cyrillic ("4CC" for "ЧСС", "Mc" for "мс").
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ecgscan.calibration.header_ocr import parse_settings

_I = re.IGNORECASE

PHI_LABELS = re.compile(
    r"(?:пациент|patient|bemor|ф\.?\s*и\.?\s*о\b|фио\b|фамилия|имя\b|отчество|\bname\b|f\.?\s*i\.?\s*sh|"
    r"\bid\b|№\s*карт|карта\s*№|med\.?\s*rec|mrn|дата\s*рожд|д\.\s*р\.|\bdob(?![a-zа-я])|birth|tug\W{0,3}ilgan|"
    r"возраст|\bage\b|yosh|\bпол\b|\bsex\b|gender|jinsi|адрес|address|manzil|тел\.?|phone)", _I)
_NAME_LIKE = re.compile(r"^(?:[A-ZА-ЯЁЎҚҒҲ][\w'’-]+\.?\s+){1,3}[A-ZА-ЯЁЎҚҒҲ][\w'’-]+\.?$")

_N = r"(\d{2,3})"
_SEP = r"\s*[:=.]?\s*"                       # "ЧСС: 72", "ЧСС. 72", "ЧСС 72"
_MS = r"\s*[mм][sсc]"                        # ms, мс, and the OCR's mixes "мC", "mc"
_O = r"[OoОо0]"                               # O, о, or the OCR's 0 in a label
HR_RE = re.compile(r"(?:[Ч4][СC][СC]|\bHR\b|vent\.?\s*rate|heart\s*rate|\bpulse\b|пульс)" + _SEP
                   + r"(?:\d\s+)?" + _N, _I)        # a stray digit before the value ("ЧСС: 1  106")
PR_RE = re.compile(r"\b(?:[PР]-?[RQ])(?:\s*int(?:erval)?)?" + _SEP + _N + r"(?!\s*-)", _I)
QRS_RE = re.compile(r"\bQRS(?:\s*dur(?:ation)?)?[^\d\n]{0,8}?" + _N + _MS, _I)
QTQTC_RE = re.compile(r"\bQT\s*/\s*QT[cс]?" + _SEP + r"(?:\d{3}\s+)?(\d{3})\s*[/XxХх×]\s*(\d{3})", _I)
QT_RE = re.compile(r"\bQT" + _SEP + r"(\d{3})\b(?!\s*/)", _I)
QTC_RE = re.compile(r"\bQT[cс]" + _SEP + r"(\d{3})", _I)
AXIS_RE = re.compile(r"(?:" + _O + r"[cс][ьb](?:\s*Э" + _O + r"[CС])?|Э" + _O + r"[CС]|QRS\s*axis|\baxis\b)" + _SEP
                     + r"\|?\s*([-−–]?\s?\d{1,3})\s*°?", _I)
AXES3_RE = re.compile(r"P-?R-?T\s*(?:ax[eé]s)?\S*" + _SEP + r"([-−]?\d{1,3})\s*/?\s*([-−]?\d{1,3})\s*/?\s*([-−]?\d{1,3})", _I)
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?:\s*(\d{1,2})[:.](\d{2}))?")
BAND_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)\s*(?:Гц|Hz|Ги|r[uц])", _I)
HP_RE = re.compile(r"(?:ФВЧ|HPF?|high\s*pass)\s*[:=]?\s*(\d+(?:[.,]\d+)?)", _I)
LP_RE = re.compile(r"(?:ФНЧ|LPF?|low\s*pass)\s*[:=]?\s*(\d+(?:[.,]\d+)?)", _I)
MAINS_RE = re.compile(r"(?:~\s*|сеть\s*|AC\s*|notch\s*)(50|60)\s*(?:Гц|Hz)|\b(50|60)\s*(?:Гц|Hz)\s*(?:сеть|AC)?(?!\s*-)", _I)
_FIELD_HINT = re.compile(r"(?:\d\s*(?:мс|ms|mc|уд|bpm|мм|mm|Гц|Hz|°))|ЧСС|QRS|QT|\bPR\b|\bPQ\b", _I)


@dataclass
class DeviceMeasurements:
    hr_bpm: int | None = None
    pr_ms: int | None = None
    qrs_ms: int | None = None
    qt_ms: int | None = None
    qtc_ms: int | None = None
    axis_deg: int | None = None
    p_axis_deg: int | None = None
    t_axis_deg: int | None = None

    def any(self) -> bool:
        return any(v is not None for v in asdict(self).values())


@dataclass
class HeaderFields:
    speeds: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    measurements: DeviceMeasurements = field(default_factory=DeviceMeasurements)
    filter_hp_hz: float | None = None
    filter_lp_hz: float | None = None
    mains_hz: int | None = None
    recorded_at: str | None = None              # ISO 8601, minutes; the recording's, not the birth date
    device_model: str | None = None
    conclusion: str | None = None               # the recorder's; never used as ours
    phi_lines_dropped: int = 0
    safe_lines: list[str] = field(default_factory=list)   # what may be kept: no personal data


_DIGIT_IN_WORD = re.compile(r"(?<=[^\W\d_])0|0(?=[^\W\d_])")   # "D0B": the OCR's 0 for O in a word


def is_phi(line: str) -> bool:
    """A line with a personal-data label (read through the OCR's 0-for-O), a date without a time
    (a birth date; the recording's date has its time: S3 bench, "D0B: 16.07.1979" leaked), or an
    unlabelled line that looks like a name."""
    s = line.strip()
    if PHI_LABELS.search(_DIGIT_IN_WORD.sub("o", s)):
        return True
    if any(not d.group(4) for d in DATE_RE.finditer(s)):
        return True
    return not any(c.isdigit() for c in s) and bool(_NAME_LIKE.match(s)) and not s.endswith(".") \
        and s.upper() != s


def _num(v: str) -> float:
    return float(v.replace(",", ".").replace("−", "-").replace("–", "-").replace(" ", ""))


def _first(rx: re.Pattern, lines: list[str], lo: int, hi: int) -> int | None:
    for ln in lines:
        for m in rx.finditer(ln):
            v = int(_num(m.group(1)))
            if lo <= v <= hi:
                return v
    return None


def parse_fields(lines: list[str], *, models: list[str] | None = None) -> HeaderFields:
    """``lines``: OCR lines, top to bottom. ``models``: recorder-model patterns (regex, from config)."""
    keep = [ln for ln in lines if not is_phi(ln)]
    f = HeaderFields(phi_lines_dropped=len(lines) - len(keep), safe_lines=keep)
    f.speeds, f.gains = parse_settings(keep)
    m = f.measurements
    m.hr_bpm = _first(HR_RE, keep, 20, 300)
    m.pr_ms = _first(PR_RE, keep, 60, 400)
    m.qrs_ms = _first(QRS_RE, keep, 40, 250)
    for ln in keep:
        q = QTQTC_RE.search(ln)
        if q:
            m.qt_ms, m.qtc_ms = int(q.group(1)), int(q.group(2))
            break
    m.qt_ms = m.qt_ms or _first(QT_RE, keep, 200, 700)
    m.qtc_ms = m.qtc_ms or _first(QTC_RE, keep, 200, 700)
    for ln in keep:
        a3 = AXES3_RE.search(ln)
        if a3:
            m.p_axis_deg, m.axis_deg, m.t_axis_deg = (int(_num(g)) for g in a3.groups())
            break
    if m.axis_deg is None:
        for ln in keep:
            a = AXIS_RE.search(ln)
            if a and -180 <= int(_num(a.group(1))) <= 360:
                m.axis_deg = int(_num(a.group(1)))
                break
    for ln in keep:
        b = BAND_RE.search(ln)
        if b:
            f.filter_hp_hz, f.filter_lp_hz = _num(b.group(1)), _num(b.group(2))
        hp, lp = HP_RE.search(ln), LP_RE.search(ln)
        if hp:
            f.filter_hp_hz = _num(hp.group(1))
        if lp:
            f.filter_lp_hz = _num(lp.group(1))
        mn = MAINS_RE.search(ln)
        if mn and f.mains_hz is None:
            f.mains_hz = int(mn.group(1) or mn.group(2))
        d = DATE_RE.search(ln)
        if d and f.recorded_at is None and d.group(4):
            day, mon, yr, hh, mi = (int(g) for g in d.groups())
            if 1 <= day <= 31 and 1 <= mon <= 12 and 0 <= hh < 24 and 0 <= mi < 60:
                f.recorded_at = f"{yr:04d}-{mon:02d}-{day:02d}T{hh:02d}:{mi:02d}"
    for pat in models or []:
        for ln in keep:
            mm = re.search(pat, ln, _I)
            if mm:
                f.device_model = mm.group(0).strip()
                break
        if f.device_model:
            break
    concl = [_tidy(ln) for ln in keep if _conclusion_like(ln, f.device_model)]
    f.conclusion = " ".join(concl) or None
    f.device_model = _tidy(f.device_model) if f.device_model else None
    return f


def _tidy(text: str) -> str:
    """Single spaces, none before punctuation (the OCR's "брадикардия ." and doubled gaps)."""
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(text.split()))


def _conclusion_like(line: str, model: str | None) -> bool:
    """A free-text line: mostly letters, no measurement, setting, date or model in it."""
    s = line.strip()
    if len(s) < 6 or _FIELD_HINT.search(s) or DATE_RE.search(s) or (model and model in s):
        return False
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) >= 0.6
