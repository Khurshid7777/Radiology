"""S4 (TZ 18.14): the report PDF opens in PDFium, Poppler and MuPDF, every page renders, every
flag is found as text in Russian and Uzbek, and the fonts text is set in are embedded."""

import shutil

import numpy as np
import pytest

from ecgscan.bench.pdf_check import check
from ecgscan.contract.models import DeviceMeasurements, HeaderOcr, Trace
from ecgscan.report_block import render_rows, scanner_rows

pytest.importorskip("pypdfium2")
pytest.importorskip("pymupdf")
pytestmark = pytest.mark.skipif(not shutil.which("pdftoppm") or not shutil.which("pdffonts"),
                                reason="poppler-utils not installed")


def _read(result_factory, synth_mv):
    leads = {lead: Trace.from_array(mv[int(2.5 * (i // 3) * 500):int(2.5 * (i // 3 + 1) * 500)],
                                    start_s=2.5 * (i // 3), fs=500.0, lead=lead, quality=0.9 if i != 4 else 0.55)
             for i, (lead, mv) in enumerate(synth_mv.items())}
    rhythm = [Trace.from_array(synth_mv["II"], start_s=0.0, fs=500.0, lead="II", quality=0.9)]
    ho = HeaderOcr(engine="rapidocr:cyrillic_PP-OCRv5_rec_mobile", device_model="EK-12",
                   device_measurements=DeviceMeasurements(hr_bpm=72, qrs_ms=92, axis_deg=45, uncertain=["axis_deg"]))
    return result_factory(leads=leads, rhythm=rhythm, status="needs_confirmation",
                          warnings=["SPEED_SOURCES_DISAGREE", "FADED_PAPER"], header_ocr=ho)


def _expected(res):
    return [s for lang in ("ru", "uz") for row in scanner_rows(res, lang) for s in row]


def test_the_flags_table_reads_back_in_every_engine(tmp_path, result_factory, synth_mv):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate

    from cardiolens.outputs.pdf_report import _register_font

    res = _read(result_factory, synth_mv)
    body = ParagraphStyle("b", fontName=_register_font(), fontSize=8, leading=11)
    pdf = tmp_path / "flags.pdf"
    SimpleDocTemplate(str(pdf)).build([render_rows(scanner_rows(res, lang), body, width=17 * cm) for lang in ("ru", "uz")])
    out = check(pdf, _expected(res))
    assert out["ok"], {k: v.get("missing", v) for k, v in out["engines"].items()}
    assert {f["name"].split("+")[-1] for f in out["fonts"] if f["used"]} == {"DejaVuSans"}


def test_the_core_report_of_a_scan_opens_everywhere(tmp_path, result_factory, synth_mv):
    """The doctor's PDF, written by the core from a scanned page as the service sends it, with
    every flag in both languages (``Provenance.report_rows``, core D184)."""
    import cv2

    from cardiolens.outputs.pdf_report import OVERLAY_NOTE_UZ, write_pdf
    from cardiolens.pipeline import run_pipeline
    from ecgscan.contract.cardiolens_adapter import to_signal_representation

    res = _read(result_factory, synth_mv)
    sig = to_signal_representation(res)
    sig.provenance.overlay_jpeg = cv2.imencode(".jpg", np.full((200, 400, 3), 230, np.uint8))[1].tobytes()
    pdf = tmp_path / "report.pdf"
    write_pdf(run_pipeline(sig), str(pdf))
    expect = [OVERLAY_NOTE_UZ[:50], "Заключение (RU)", "Xulosa (UZ)", *_expected(res)]
    out = check(pdf, expect)
    assert out["ok"], {k: v.get("missing", v) for k, v in out["engines"].items()}
