"""D183: the recorder's printed measurements (scanner S3 header read) against ours."""

from types import SimpleNamespace as NS

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import device_mismatches, run_pipeline


def test_a_rate_the_recorder_printed_differently_is_said_and_does_not_gate():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.device_measurements = {"hr_bpm": 90}
    res = run_pipeline(sig, mode="A")
    f = next(f for f in res.findings.findings if f.code == "device_mismatch")
    assert "ЧСС" in f.label_ru and "аппарат 90" in f.label_ru
    assert f.group == "quality"
    assert res.measurements.heart_rate_bpm is not None          # ours stay: advisory only
    assert "photo_unreliable" not in res.findings.codes()


def test_values_within_the_tolerance_are_not_a_mismatch():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    hr = round(run_pipeline(sig, mode="A").measurements.heart_rate_bpm)
    sig.provenance.device_measurements = {"hr_bpm": hr + 4}
    assert "device_mismatch" not in run_pipeline(sig, mode="A").findings.codes()


def test_tolerances_follow_tz_18_6():
    m = NS(heart_rate_bpm=70.0, intervals=NS(pr_ms=160.0, qrs_ms=90.0, qt_ms=400.0))
    dev = {"hr_bpm": 76, "pr_ms": 175, "qrs_ms": 106, "qt_ms": 420}
    assert [k for k, _, _ in device_mismatches(m, dev)] == ["hr_bpm", "qrs_ms"]


def test_the_scanner_header_reaches_the_signal_without_the_recorders_text():
    from cardiolens.telegram.analyze import attach_device_header

    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    dm = NS(model_dump=lambda: {"hr_bpm": 75, "pr_ms": None, "qrs_ms": 96, "axis_deg": 743 % 360,
                                "uncertain": ["axis_deg"]})
    h = NS(device_measurements=dm, device_model="SCHILLER AT-102", filter_hp_hz=0.05, filter_lp_hz=None,
           mains_hz=50, device_text_conclusion="Синусовый ритм")
    attach_device_header(sig, NS(header_ocr=h))
    assert sig.provenance.device_measurements == {"hr_bpm": 75, "qrs_ms": 96}
    assert sig.provenance.device_info == {"device_model": "SCHILLER AT-102", "filter_hp_hz": 0.05, "mains_hz": 50}
    assert "Синусовый" not in sig.provenance.model_dump_json()


def test_the_recorder_line_for_the_page_and_the_pdf():
    from cardiolens.outputs.pdf_report import device_line, measurement_line

    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.device_info = {"device_model": "ЭК12Т-01-Р-Д", "filter_hp_hz": 0.05, "filter_lp_hz": 150.0,
                                  "mains_hz": 50}
    res = run_pipeline(sig, mode="A")
    line = device_line(res)
    assert line == "Аппарат (по шапке листа): ЭК12Т-01-Р-Д · фильтр 0.05–150.0 Гц · сеть 50 Гц"
    assert line in measurement_line(res)
    sig.provenance.device_info = None
    assert device_line(run_pipeline(sig, mode="A")) is None


def test_pdf_has_both_languages_the_scanner_rows_and_the_mismatch(tmp_path):
    """D184 (S4, TZ 18.10): RU and UZ measurement lines carry the recorder mismatch, and the
    PDF builds with the scanner's rows per language (text is checked by the scanner's PDF
    check across engines)."""
    from pathlib import Path

    from cardiolens.outputs.pdf_report import measurement_line, write_pdf

    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.device_measurements = {"hr_bpm": 95}
    sig.provenance.report_rows = {"ru": [("Источник", "фото листа")], "uz": [("Manba", "varaq surati")]}
    res = run_pipeline(sig, mode="A")
    ru, uz = measurement_line(res, "ru"), measurement_line(res, "uz")
    assert "ЧСС" in ru and "аппарат 95" in ru
    assert "YuQS" in uz and "apparat 95" in uz and "ЧСС" not in uz
    path = write_pdf(res, str(tmp_path / "r.pdf"))
    assert Path(path).stat().st_size > 10_000
