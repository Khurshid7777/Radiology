"""The scanner's section of the PDF report (S4, TZ 18.10): what the doctor needs to know about
how the signal was read off the photo, in Russian and Uzbek.

``scanner_rows(result, lang)`` gives (label, value) pairs, already translated
(``configs/scanner/i18n``, keys ``report.*``):

* the source (photo, scan, PDF), the client and how the patient data were covered;
* the reading's status and whether a doctor confirmed it;
* paper speed and gain, each with where it came from;
* the lead layout and its confidence, the number of sheets;
* the overall quality, the leads read below the overlay's "green" quality and the leads not read;
* every warning in words;
* the recorder's own measurements from the header, values no check vouches for marked "?".

Nothing personal goes in: the header fields kept by S3 are the recorder's (SD-091). The core's
PDF renders the rows (``render_rows``); the recorder-vs-ours comparison (``device_mismatch``) is
the core's.
"""

from __future__ import annotations

from ecgscan.artifacts import QUALITY_COLOURS
from ecgscan.config import load_yaml
from ecgscan.contract.models import STANDARD_LEADS, ScanResult
from ecgscan.i18n import t

LEAD_OK_QUALITY = QUALITY_COLOURS[0][0]      # the overlay's green: the same cut on the page and in the PDF


def _num(v: float, lang: str) -> str:
    s = f"{v:g}"
    return s.replace(".", ",") if lang == "ru" else s


def _scale(value, source: str | None, kind: str, lang: str) -> str:
    if value is None:
        return t("report.unknown", lang)
    src = t(f"report.src.{source}", lang) if source else t("report.src.default", lang)
    return t(f"report.{kind}_value", lang, value=_num(value, lang), source=src)


def _device(result: ScanResult, lang: str) -> list[tuple[str, str]]:
    ho = result.header_ocr
    if ho is None:
        return []
    rows: list[tuple[str, str]] = []
    if ho.device_model:
        rows.append((t("report.device_model", lang), ho.device_model))
    if ho.recorded_at:
        rows.append((t("report.recorded_at", lang), ho.recorded_at))
    m = ho.device_measurements
    if m is None:
        return rows
    unsure = set(m.uncertain)

    def mark(field: str) -> str:
        return "?" if field in unsure else ""

    parts = []
    if m.hr_bpm is not None:
        parts.append(t("report.hr", lang, v=f"{m.hr_bpm}{mark('hr_bpm')}"))
    for name, field in (("PR", "pr_ms"), ("QRS", "qrs_ms"), ("QT", "qt_ms"), ("QTc", "qtc_ms")):
        v = getattr(m, field)
        if v is not None:
            parts.append(t("report.ms", lang, name=name, v=f"{v}{mark(field)}"))
    if m.axis_deg is not None:
        parts.append(t("report.axis", lang, v=m.axis_deg) + mark("axis_deg"))        # -32°?, not -32?°
    if parts:
        value = ", ".join(parts)
        if unsure:
            value += ". " + t("report.device_note", lang)
        rows.append((t("report.device", lang), value))
    return rows


def scanner_rows(result: ScanResult, lang: str = "ru") -> list[tuple[str, str]]:
    src, cal = result.source, result.calibration
    rows = [(t("report.source", lang), t(f"report.source.{src.modality}", lang))]
    if src.client:
        rows.append((t("report.client", lang), t(f"report.client.{src.client}", lang)))
    # The picture in the report is the server's overlay: its header band is blanked whatever the
    # client did (D136, ``artifacts.mask_bands``); the client's own masking is named when there was one.
    method = src.capture.mask_method if src.capture else None
    if method and method != "none":
        mask = t(f"report.mask.{method}", lang)
    elif src.header_masked or load_yaml("pipeline")["artifacts"]["mask_bands"]:
        mask = t("report.mask.server", lang)
    else:
        mask = t("report.mask.none", lang)
    rows.append((t("report.mask", lang), mask))
    status = t(f"status.{result.status}", lang)
    if result.confirmation is not None:
        status += f" ({t('report.confirmed', lang)})"
    rows.append((t("report.status", lang), status))
    rows.append((t("report.speed", lang), _scale(cal.paper_speed_mm_s, cal.speed_source, "speed", lang)))
    rows.append((t("report.gain", lang), _scale(cal.gain_mm_mv, cal.gain_source, "gain", lang)))
    lay = result.layout
    layout = t("report.layout_value", lang, layout=lay.name, conf=_num(round(lay.confidence, 2), lang))
    if lay.confirmed_by_user:
        layout += f" ({t('report.confirmed', lang)})"
    rows.append((t("report.layout", lang), layout))
    if result.pages and len(result.pages) > 1:
        rows.append((t("report.pages", lang), str(len(result.pages))))
    rows.append((t("report.quality", lang), _num(round(result.quality.overall, 2), lang)))
    if result.leads:
        weak = [f"{lead} {_num(round(tr.quality, 2), lang)}" for lead, tr in result.leads.items()
                if tr.quality < LEAD_OK_QUALITY]
        missing = [lead for lead in STANDARD_LEADS if lead not in result.leads]
        parts = []
        if weak:
            parts.append(t("report.leads_weak", lang, limit=_num(LEAD_OK_QUALITY, lang), leads=", ".join(weak)))
        if missing:
            parts.append(t("report.leads_missing", lang, leads=", ".join(missing)))
        rows.append((t("report.leads", lang), "; ".join(parts) or t("report.none", lang)))
    warnings = [t(f"warning.{w}", lang, ppm=_num(round(result.quality.px_per_mm or 0, 1), lang))
                for w in result.warnings]
    rows.append((t("report.warnings", lang), " ".join(warnings) or t("report.none", lang)))
    rows += _device(result, lang)
    return rows


def render_rows(rows: list[tuple[str, str]], body, *, width: float):
    """A reportlab table of (label, value) rows in ``body``'s font, ``width`` points wide."""
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    data = [[Paragraph(esc(k), body), Paragraph(esc(v), body)] for k, v in rows]
    table = Table(data, colWidths=[0.3 * width, 0.7 * width])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table
