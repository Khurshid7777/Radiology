"""PDF report (TZ §11.5): summary + waveform + causal graph + conclusion + disclaimer.

Uses a Cyrillic-capable TrueType font (DejaVuSans, shipped with matplotlib) so
Russian and Uzbek text render correctly in the PDF.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cardiolens.outputs.conclusion import format_conclusion
from cardiolens.outputs.graph_viz import render_causal_graph
from cardiolens.outputs.waveform import render_waveform
from cardiolens.pipeline import PipelineResult

_FONT = "DejaVuSans"

# A photo read by the scanner (D163): what the overlay shows and how to read it, RU and UZ.
OVERLAY_NOTE_RU = ("Источник — фото бумажной ЭКГ, оцифровано. Цветные линии — трассы, прочитанные с листа: "
                   "зелёный — уверенно, оранжевый — сомнительно, красный — плохо; число у отведения — качество "
                   "чтения. Если линии не лежат на кривых, заключению верить нельзя — переснимите лист. "
                   "Шапка с данными пациента закрыта.")
OVERLAY_NOTE_UZ = ("Manba — qog‘oz EKG surati, raqamlashtirilgan. Rangli chiziqlar — varaqdan o‘qilgan chiziqlar: "
                   "yashil — ishonchli, to‘q sariq — shubhali, qizil — yomon; tarmoq yonidagi son — o‘qish sifati. "
                   "Chiziqlar EKG egri chiziqlari ustida yotmasa, xulosaga ishonib bo‘lmaydi — varaqni qayta "
                   "suratga oling. Bemor ma’lumotlari bo‘lgan sarlavha yopilgan.")



UNRELIABLE_CODES = frozenset({"photo_unreliable", "no_ecg_signal", "amplitude_implausible"})


_SPEED_KNOWN = ("(header_ocr)", "(user)")


def device_line(result, lang: str = "ru") -> str | None:
    """The recorder as its printed header names it (scanner S3, D183): model, filters, mains.
    Its own measurements are not repeated here: they appear only where they differ from ours
    (``device_mismatch``)."""
    info = result.signal.provenance.device_info or {}
    if not info:
        return None
    uz = lang == "uz"
    parts = [str(info["device_model"])] if info.get("device_model") else []
    hp, lp = info.get("filter_hp_hz"), info.get("filter_lp_hz")
    if hp is not None or lp is not None:
        parts.append(f"{'filtr' if uz else 'фильтр'} {hp if hp is not None else '—'}–"
                     f"{lp if lp is not None else '—'} {'Gs' if uz else 'Гц'}")
    if info.get("mains_hz"):
        parts.append(f"{'tarmoq' if uz else 'сеть'} {info['mains_hz']} {'Gs' if uz else 'Гц'}")
    if not parts:
        return None
    return ("Apparat (varaq sarlavhasidan): " if uz else "Аппарат (по шапке листа): ") + " · ".join(parts)


def speed_assumed_note(result, lang: str = "ru") -> str | None:
    """A photo whose paper speed was neither read from the strip nor given by the user: the
    time scale is the 25 mm/s default (D180). A strip printed at 50 mm/s read so halves the
    rate — said next to the numbers, with what to do."""
    prov = result.signal.provenance
    if not prov.digitized:
        return None
    notes = prov.notes
    if any(n.startswith("scale given by the user") for n in notes):
        return None
    cal = next((n for n in notes if n.startswith("calibration: speed=")), None)
    if cal is not None and any(k in cal.split(" gain=")[0] for k in _SPEED_KNOWN):
        return None
    if lang == "uz":
        return ("Lenta tezligi suratdan o‘qilmadi — 25 mm/s deb olindi. Agar lentada 50 mm/s yozilgan bo‘lsa, "
                "uni «Lenta tezligi» maydonida tanlang va tahlilni takrorlang: aks holda YuUS ikki baravar past, "
                "intervallar ikki baravar uzun chiqadi.")
    return ("Скорость ленты не прочитана со снимка — принята 25 мм/с. Если на ленте напечатано 50 мм/с, "
            "выберите её в поле «Скорость ленты» и повторите анализ: иначе ЧСС будет вдвое ниже, интервалы вдвое длиннее.")


def measurement_line(result, lang: str = "ru") -> str:
    """The report's measurement line (RU or UZ, TZ 18.10). A record the photo gates call
    unreliable carries no number a doctor could take for a measurement (D178; the narrative
    withholds them, D128). The recorder's line and a mismatch with it follow (D183)."""
    uz = lang == "uz"
    if result.findings.codes() & UNRELIABLE_CODES:
        if uz:
            return ("O‘lchovlar keltirilmaydi: surat ishonchsiz o‘qildi — undan olingan raqamlar xato bo‘lishi "
                    "mumkin. EKGni qayta suratga oling.")
        return ("Измерения не приводятся: снимок распознан ненадёжно — числа, считанные с него, "
                "могут быть ошибочными. Переснимите ЭКГ.")
    m = result.measurements

    def n(v) -> str:            # a measurement that was not made prints as a dash, not "None"
        return "—" if v is None else f"{v:g}" if isinstance(v, float) else str(v)

    ms, ivs = ("ms", m.intervals) if uz else ("мс", m.intervals)
    line = (f"{'YuUS' if uz else 'ЧСС'}: {n(m.heart_rate_bpm)} {'/daq' if uz else '/мин'} · PR: {n(ivs.pr_ms)} {ms} · "
            f"QRS: {n(ivs.qrs_ms)} {ms} · QTc(F/B): {n(ivs.qtc_fridericia_ms)}/{n(ivs.qtc_bazett_ms)} {ms} · "
            f"{'QRS o‘qi' if uz else 'ось QRS'}: {n(m.axes.qrs_deg)}°")
    mismatch = next((f for f in result.findings.findings if f.code == "device_mismatch"), None)
    for extra in (speed_assumed_note(result, lang), device_line(result, lang),
                  (mismatch.label_uz if uz else mismatch.label_ru) if mismatch else None):
        if extra is not None:
            line += f"<br/><i>{_esc(extra)}</i>"
    return line


def _register_font() -> str:
    try:
        import matplotlib

        base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont(_FONT, str(base / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont(_FONT + "-Bold", str(base / "DejaVuSans-Bold.ttf")))
        return _FONT
    except Exception:
        return "Helvetica"  # fallback (Cyrillic may not render)


def write_pdf(result: PipelineResult, path: str) -> str:
    font = _register_font()
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_"))
    wave_png = str(tmp / "wave.png")
    graph_png = str(tmp / "graph.png")
    render_waveform(result, wave_png)
    render_causal_graph(result, graph_png)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=font,
                          fontSize=8, leading=11)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font + "-Bold"
                        if font != "Helvetica" else "Helvetica-Bold", fontSize=14)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font + "-Bold"
                        if font != "Helvetica" else "Helvetica-Bold", fontSize=10)

    doc = SimpleDocTemplate(path, pagesize=A4, title="CardioVision ECG Report")
    flow = [Paragraph("CardioVision — заключение по ЭКГ / EKG bo‘yicha xulosa", h1), Spacer(1, 0.3 * cm)]

    flow.append(Paragraph(measurement_line(result, "ru"), body))
    flow.append(Spacer(1, 0.15 * cm))
    flow.append(Paragraph(measurement_line(result, "uz"), body))
    flow.append(Spacer(1, 0.3 * cm))

    flow.append(Paragraph("Кривая ЭКГ", h2))
    flow.append(Image(wave_png, width=17 * cm, height=17 * cm * 0.5))
    flow.append(Spacer(1, 0.2 * cm))

    overlay = result.signal.provenance.overlay_jpeg
    if overlay:
        # A photo read by the scanner: the traces drawn over the page, as the doctor's
        # confirmation screen shows them (TZ scanner §6.3; D163).
        from PIL import Image as PILImage

        ov_path = tmp / "overlay.jpg"
        ov_path.write_bytes(overlay)
        with PILImage.open(ov_path) as im:
            w, h = im.size
        # Heading, notes and picture stay on one page (the picture is most of a page).
        flow.append(KeepTogether([
            Paragraph("Снимок: прочитанные трассы / Suratdagi o‘qilgan chiziqlar", h2),
            Paragraph(_esc(OVERLAY_NOTE_RU), body),
            Paragraph(_esc(OVERLAY_NOTE_UZ), body),
            Image(str(ov_path), width=17 * cm, height=17 * cm * h / max(1, w)),
        ]))
        flow.append(Spacer(1, 0.2 * cm))

    rows = result.signal.provenance.report_rows or {}
    if rows:
        # The scanner's own account of the read (S4, TZ 18.10, D184): source, status and
        # warnings, scale and its sources, layout, per-lead quality — per language.
        heading = Paragraph("Оцифровка снимка / Surat raqamlashtirilishi", h2)
        for lang in ("ru", "uz"):
            data = [[Paragraph(_esc(a), body), Paragraph(_esc(b), body)] for a, b in rows.get(lang, [])]
            if not data:
                continue
            t = Table(data, colWidths=[6 * cm, 11 * cm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
            flow.append(KeepTogether([heading, t]) if heading is not None else t)
            flow.append(Spacer(1, 0.2 * cm))
            heading = None

    flow.append(Paragraph("Причинно-следственный граф", h2))
    flow.append(Image(graph_png, width=15 * cm, height=15 * cm * 0.55))
    flow.append(Spacer(1, 0.2 * cm))

    flow.append(Paragraph("Заключение (RU)", h2))
    for line in format_conclusion(result, "ru").split("\n"):
        flow.append(Paragraph(_esc(line) or "&nbsp;", body))
    flow.append(Spacer(1, 0.3 * cm))

    flow.append(Paragraph("Xulosa (UZ)", h2))
    for line in format_conclusion(result, "uz").split("\n"):
        flow.append(Paragraph(_esc(line) or "&nbsp;", body))

    doc.build(flow)
    # Remove the whole intermediate-image temp dir, not just its files.
    shutil.rmtree(tmp, ignore_errors=True)
    return path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
