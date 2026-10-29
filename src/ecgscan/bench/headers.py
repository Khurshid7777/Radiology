"""Synthetic printed ECG headers with ground truth (S3, TZ 18.6, 18.14).

A header is a few lines of text above the traces, as recorders print them: patient lines
(name, ID, birth date), the recorder's own measurements, the settings (speed, gain, filters),
the date and model, and the recorder's text conclusion. Styles follow the devices in use in
Uzbekistan: Russian-labelled (Axion / Alton type), English-labelled (Schiller / Mindray / Edan
type), and Uzbek labels for the patient lines. Every line keeps its pixel box and whether it
carries personal data, so masking can be checked against the truth.

Renders with system fonts (DejaVu, Liberation, FreeSans); a font that is not installed is
skipped. Deterministic by seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/freefont")
FONTS = ("DejaVuSans.ttf", "DejaVuSansMono.ttf", "LiberationSans-Regular.ttf", "LiberationMono-Regular.ttf",
         "FreeSans.ttf", "DejaVuSerif.ttf")

SURNAMES = ("Каримов", "Иванова", "Юсупов", "Рахимова", "Ахмедов", "Петров", "Тошматова", "Абдуллаев")
NAMES = ("Акмал", "Дилноза", "Сергей", "Гульнора", "Бахтиёр", "Ольга", "Шерзод", "Мадина")
PATRONYMICS = ("Рустамович", "Алишеровна", "Иванович", "Бахтияровна", "Тимурович", "Петровна")
LAT = str.maketrans({"А": "A", "а": "a", "Б": "B", "б": "b", "В": "V", "в": "v", "Г": "G", "г": "g", "Д": "D",
                     "д": "d", "Е": "E", "е": "e", "Ж": "J", "ж": "j", "З": "Z", "з": "z", "И": "I", "и": "i",
                     "Й": "Y", "й": "y", "К": "K", "к": "k", "Л": "L", "л": "l", "М": "M", "м": "m", "Н": "N",
                     "н": "n", "О": "O", "о": "o", "П": "P", "п": "p", "Р": "R", "р": "r", "С": "S", "с": "s",
                     "Т": "T", "т": "t", "У": "U", "у": "u", "Ф": "F", "ф": "f", "Х": "X", "х": "x", "Ч": "Ch",
                     "ч": "ch", "Ш": "Sh", "ш": "sh", "Ю": "Yu", "ю": "yu", "Я": "Ya", "я": "ya", "ё": "yo",
                     "ы": "i", "ь": "", "ъ": ""})
MODELS_RU = ("ЭК12Т-01-Р-Д", "Аксион ЭК1Т-1/3-07", "Альтон-106", "ЭК3Т-01-Р-Д")
MODELS_EN = ("SCHILLER AT-102", "Mindray BeneHeart R12", "EDAN SE-1201", "Nihon Kohden ECG-2350")
CONCL_RU = ("Синусовый ритм. Нормальная ЭКГ.", "Синусовая тахикардия.", "Синусовая брадикардия.",
            "Ритм синусовый. Отклонение ЭОС влево.", "Фибрилляция предсердий.")
CONCL_EN = ("Sinus rhythm. Normal ECG.", "Sinus tachycardia.", "Sinus bradycardia.", "Left axis deviation.",
            "Atrial fibrillation.")


@dataclass
class HeaderLine:
    text: str
    box: tuple[int, int, int, int]          # x0, y0, x1, y1 in the image
    phi: bool                               # carries personal data (name, ID, birth date)


@dataclass
class HeaderCase:
    case_id: str
    style: str                              # ru | en | uz
    image: np.ndarray                       # BGR
    px_per_mm: float
    truth: dict                             # speed, gain, hr, pr, qrs, qt, qtc, axis, recorded_at, model, conclusion, phi
    lines: list[HeaderLine] = field(default_factory=list)
    degrade: dict = field(default_factory=dict)


def _fonts() -> list[str]:
    out = []
    for name in FONTS:
        for d in FONT_DIRS:
            p = Path(d) / name
            if p.exists():
                out.append(str(p))
                break
    return out


def _truth(rng: np.random.Generator, style: str) -> dict:
    hr = int(rng.integers(45, 130))
    pr = int(rng.integers(120, 220)) if rng.random() > 0.1 else None   # AF: no PR
    qrs = int(rng.integers(70, 130))
    qt = int(rng.integers(320, 460))
    qtc = int(round(qt / np.sqrt(60.0 / hr)))
    axis = int(rng.integers(-40, 110))
    p_axis, t_axis = int(rng.integers(0, 80)), int(rng.integers(0, 90))
    day, mon, yr = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(2023, 2027))
    hh, mm = int(rng.integers(7, 19)), int(rng.integers(0, 60))
    sur, name = SURNAMES[rng.integers(len(SURNAMES))], NAMES[rng.integers(len(NAMES))]
    pat = PATRONYMICS[rng.integers(len(PATRONYMICS))]
    pid = f"{int(rng.integers(10**6, 10**7))}"
    dob = f"{int(rng.integers(1, 29)):02d}.{int(rng.integers(1, 13)):02d}.{int(rng.integers(1940, 2015))}"
    return dict(
        speed=float(rng.choice([25, 50], p=[0.8, 0.2])), gain=float(rng.choice([5, 10, 20], p=[0.1, 0.8, 0.1])),
        hr=hr, pr=pr, qrs=qrs, qt=qt, qtc=qtc, axis=axis, p_axis=p_axis, t_axis=t_axis,
        recorded_at=f"{yr:04d}-{mon:02d}-{day:02d}T{hh:02d}:{mm:02d}", date_text=f"{day:02d}.{mon:02d}.{yr} {hh:02d}:{mm:02d}",
        model=(MODELS_EN if style == "en" else MODELS_RU)[rng.integers(4)],
        conclusion=(CONCL_EN if style == "en" else CONCL_RU)[rng.integers(5)],
        hp=float(rng.choice([0.05, 0.5])), lp=float(rng.choice([35, 100, 150])),
        phi=dict(name=f"{sur} {name} {pat}" if style != "uz" else f"{sur} {name}".translate(LAT), id=pid, dob=dob),
    )


def _num(v: float) -> str:
    return f"{v:g}"


def _lines(t: dict, style: str, rng: np.random.Generator) -> list[tuple[str, bool]]:
    """(text, phi) lines in the device's style."""
    s, g = _num(t["speed"]), _num(t["gain"])
    sep = rng.choice([" ", ""])
    if style == "en":
        pr = f"PR int {t['pr']} ms" if t["pr"] else "PR int --- ms"
        return [
            (f"Name: {t['phi']['name'].translate(LAT)}   ID: {t['phi']['id']}", True),
            (f"DOB: {t['phi']['dob']}   Male", True),
            (f"Vent rate {t['hr']} bpm   {pr}   QRS dur {t['qrs']} ms", False),
            (f"QT/QTc {t['qt']}/{t['qtc']} ms   P-R-T axes {t['p_axis']} {t['axis']} {t['t_axis']}", False),
            (f"{s}{sep}mm/s   {g}{sep}mm/mV   {_num(t['hp'])}-{_num(t['lp'])} Hz   ~50Hz", False),
            (f"{t['date_text']}   {t['model']}", False),
            (t["conclusion"], False),
        ]
    if style == "uz":
        pr = f"PQ: {t['pr']} мс" if t["pr"] else "PQ: --- мс"
        return [
            (f"F.I.Sh.: {t['phi']['name']}   ID: {t['phi']['id']}", True),
            (f"Tug'ilgan sana: {t['phi']['dob']}", True),
            (f"ЧСС: {t['hr']} уд/мин   {pr}   QRS: {t['qrs']} мс   QT/QTc: {t['qt']}/{t['qtc']} мс", False),
            (f"Ось ЭОС: {t['axis']}°   {s}{sep}мм/с   {g}{sep}мм/мВ   ФВЧ {_num(t['hp'])} Гц   ФНЧ {_num(t['lp'])} Гц".replace(".", ","), False),
            (f"{t['date_text']}   {t['model']}", False),
            (t["conclusion"], False),
        ]
    pr = f"PQ: {t['pr']} мс" if t["pr"] else "PQ: --- мс"
    return [
        (f"Пациент: {t['phi']['name']}   ID: {t['phi']['id']}", True),
        (f"Дата рождения: {t['phi']['dob']}   Пол: М", True),
        (f"ЧСС: {t['hr']} уд/мин   {pr}   QRS: {t['qrs']} мс   QT/QTc: {t['qt']}/{t['qtc']} мс", False),
        (f"Ось: {t['axis']}°   {s}{sep}мм/с   {g}{sep}мм/мВ   {_num(t['hp'])}-{_num(t['lp'])} Гц".replace(".", ","), False),
        (f"{t['date_text']}   {t['model']}", False),
        (t["conclusion"], False),
    ]


def render_header(seed: int, *, style: str | None = None, px_per_mm: float = 10.0, text_mm: float | None = None,
                  font: str | None = None, degrade: dict | None = None) -> HeaderCase:
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    style = style or str(rng.choice(["ru", "ru", "en", "uz"]))
    fonts = _fonts()
    if not fonts:
        raise RuntimeError("no TrueType font with Cyrillic found")
    font = font or fonts[int(rng.integers(len(fonts)))]
    text_mm = text_mm or float(rng.uniform(2.2, 3.5))          # cap height on paper
    size = max(10, int(round(text_mm * px_per_mm * 1.4)))      # font size ~ 1.4 x cap height
    f = ImageFont.truetype(font, size)
    t = _truth(rng, style)
    rows = _lines(t, style, rng)
    width = int(260 * px_per_mm)
    pitch = int(size * 1.6)
    img = Image.new("RGB", (width, pitch * len(rows) + 2 * size), "white")
    d = ImageDraw.Draw(img)
    lines = []
    x0 = int(8 * px_per_mm)
    for i, (text, phi) in enumerate(rows):
        y0 = size + i * pitch
        d.text((x0, y0), text, fill="black", font=f)
        bx = d.textbbox((x0, y0), text, font=f)
        lines.append(HeaderLine(text, tuple(int(v) for v in bx), phi))
    bgr = np.asarray(img)[:, :, ::-1].copy()
    deg = dict(degrade or {})
    bgr = _degrade(bgr, deg, rng, px_per_mm)
    return HeaderCase(f"hdr-{style}-{seed}", style, bgr, px_per_mm, t, lines, deg)


def _degrade(bgr: np.ndarray, deg: dict, rng: np.random.Generator, ppm: float) -> np.ndarray:
    """grid: red|gray (printed under the text), faint (0..1), blur_px, noise, jpeg, rotate_deg."""
    import cv2

    out = bgr.astype(np.float32)
    if deg.get("grid"):
        h, w = out.shape[:2]
        line = np.array((60, 60, 200) if deg["grid"] == "red" else (150, 150, 150), np.float32)
        minor = np.array((225, 228, 253) if deg["grid"] == "red" else (225, 225, 225), np.float32)
        paper = np.full_like(out, 255)
        step = ppm
        for x in np.arange(0, w, step):
            paper[:, int(x)] = minor
        for y in np.arange(0, h, step):
            paper[int(y)] = minor
        for x in np.arange(0, w, 5 * step):
            paper[:, int(x):int(x) + 2] = line
        for y in np.arange(0, h, 5 * step):
            paper[int(y):int(y) + 2] = line
        ink = out.min(axis=2, keepdims=True) / 255.0            # 0 = ink
        out = paper * ink                                       # black text printed over the grid
    if deg.get("faint"):
        out = 255 - (255 - out) * (1 - float(deg["faint"]))
    if deg.get("rotate_deg"):
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(deg["rotate_deg"]), 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderValue=(255, 255, 255))
    if deg.get("blur_px"):
        k = int(2 * round(3 * deg["blur_px"]) + 1)
        out = cv2.GaussianBlur(out, (k, k), float(deg["blur_px"]))
    if deg.get("noise"):
        out = out + rng.normal(0, float(deg["noise"]), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if deg.get("jpeg"):
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(deg["jpeg"])])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


DEGRADES = (
    {},
    {"jpeg": 70},
    {"blur_px": 0.8, "jpeg": 80},
    {"grid": "red"},
    {"grid": "red", "jpeg": 75, "noise": 4},
    {"grid": "gray", "faint": 0.4},
    {"rotate_deg": 0.8, "noise": 6},
    {"faint": 0.55, "blur_px": 0.6},
)


def header_set(n: int, *, seed: int = 0, px_per_mm: float = 10.0) -> list[HeaderCase]:
    """``n`` headers cycling the styles' mix and the degradations above."""
    rng = np.random.default_rng(seed)
    return [render_header(int(rng.integers(2**31)), px_per_mm=px_per_mm, degrade=DEGRADES[i % len(DEGRADES)])
            for i in range(n)]
