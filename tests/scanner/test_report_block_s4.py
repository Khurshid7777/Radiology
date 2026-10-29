"""S4 (TZ 18.10): the scanner's flags for the PDF, in Russian and Uzbek, without personal data."""

import pytest

from ecgscan.contract.models import (
    Capture,
    Confirmation,
    DeviceMeasurements,
    HeaderOcr,
    Source,
    Trace,
)
from ecgscan.i18n import LANGS, t
from ecgscan.report_block import LEAD_OK_QUALITY, scanner_rows


def _photo(result_factory, **over):
    leads = {lead: Trace.from_array([0.0, 0.1, 0.0], start_s=0.0, fs=500.0, lead=lead, quality=q)
             for lead, q in (("I", 0.95), ("II", 0.62), ("III", 0.31))}
    ho = HeaderOcr(engine="rapidocr:cyrillic_PP-OCRv5_rec_mobile", device_model="Mindray BeneHeart R12",
                   recorded_at="2023-05-04T14:42",
                   device_measurements=DeviceMeasurements(hr_bpm=72, pr_ms=160, qrs_ms=92, qt_ms=380, qtc_ms=416,
                                                          axis_deg=45, uncertain=["axis_deg"]))
    base = dict(status="needs_confirmation", warnings=["SPEED_UNCERTAIN", "TRACE_GAPS"], leads=leads,
                source=Source(original_sha256="a" * 64, client="flutter", header_masked=True,
                              capture=Capture(mask_method="profile_zones")),
                header_ocr=ho)
    base.update(over)
    return result_factory(**base)


@pytest.mark.parametrize("lang", LANGS)
def test_every_flag_is_in_words(result_factory, lang):
    rows = dict(scanner_rows(_photo(result_factory), lang))
    assert rows[t("report.source", lang)] == t("report.source.photo", lang)
    assert rows[t("report.mask", lang)] == t("report.mask.profile_zones", lang)
    assert rows[t("report.status", lang)] == t("status.needs_confirmation", lang)
    assert t("report.src.user", lang) in rows[t("report.speed", lang)]
    warn = rows[t("report.warnings", lang)]
    assert t("warning.SPEED_UNCERTAIN", lang) in warn and t("warning.TRACE_GAPS", lang) in warn
    assert "SPEED_UNCERTAIN" not in warn                      # words, not codes
    leads = rows[t("report.leads", lang)]
    assert "II 0" in leads and "III 0" in leads and "I 0" not in leads.replace("II 0", "").replace("III 0", "")
    assert "aVR" in leads                                     # not read at all
    device = rows[t("report.device", lang)]
    assert "45°?" in device and "72" in device and "416" in device and "«?»" in device


def test_both_languages_give_the_same_rows(result_factory):
    ru, uz = (scanner_rows(_photo(result_factory), lang) for lang in LANGS)
    assert len(ru) == len(uz) and [v for _, v in ru] != [v for _, v in uz]


def test_where_the_scale_came_from(result_factory):
    from ecgscan.contract.models import Calibration

    res = _photo(result_factory, calibration=Calibration(paper_speed_mm_s=25, gain_mm_mv=None,
                                                         speed_source=None, confidence=0.3))
    rows = dict(scanner_rows(res, "ru"))
    assert rows["Скорость ленты"] == "25 мм/с (принято по умолчанию)"
    assert rows["Усиление"] == "не установлено"


def test_masking_and_confirmation(result_factory):
    plain = _photo(result_factory, source=Source(original_sha256="a" * 64, header_masked=False))
    assert dict(scanner_rows(plain, "uz"))[t("report.mask", "uz")] == t("report.mask.server", "uz")   # D136
    server = _photo(result_factory, source=Source(original_sha256="a" * 64, header_masked=True))
    assert dict(scanner_rows(server, "ru"))[t("report.mask", "ru")] == t("report.mask.server", "ru")
    done = _photo(result_factory, confirmation=Confirmation(by="dr", at="2026-10-29T10:00:00Z", decision="accepted"))
    assert t("report.confirmed", "ru") in dict(scanner_rows(done, "ru"))[t("report.status", "ru")]


def test_a_clean_reading_says_none(result_factory):
    res = result_factory()                                   # one lead read well, no warnings, no header
    rows = dict(scanner_rows(res, "ru"))
    assert rows["Предупреждения"] == "нет" and t("report.device", "ru") not in rows
    assert LEAD_OK_QUALITY == 0.8
