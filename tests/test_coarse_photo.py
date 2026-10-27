"""D170: a messenger-compressed photo read in the scanner's coarse mode keeps the rate and
the rhythm and withholds everything that needs a fine scale."""

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.outputs import format_conclusion, result_to_dict
from cardiolens.pipeline import run_pipeline


def _coarse(**kw):
    sig = generate_synthetic_ecg(**kw)
    sig.provenance.digitized = True
    sig.provenance.confidence = 0.6
    sig.provenance.notes += ["ecgscan scan_id=x status=needs_confirmation", "coarse: px_per_mm=3.1"]
    return sig


def test_coarse_photo_keeps_rate_and_rhythm_and_withholds_intervals():
    res = run_pipeline(_coarse(heart_rate_bpm=72, st_elevation_uv=250, st_leads=["II", "III", "aVF"]), mode="A")
    codes = res.findings.codes()
    assert "photo_coarse" in codes and "sinus_rhythm" in codes
    assert "st_elevation" not in codes and not any(f.group in ("interval", "ischemia", "hypertrophy")
                                                   for f in res.findings.findings)
    assert res.measurements.heart_rate_bpm is not None and abs(res.measurements.heart_rate_bpm - 72) < 4
    iv = res.measurements.intervals
    assert iv.pr_ms is None and iv.qrs_ms is None and iv.qt_ms is None and iv.qtc_bazett_ms is None
    assert not any(a.code == "st_elevation" for a in res.safety.alerts)
    d = result_to_dict(res)
    assert d["measurements"]["intervals"]["qrs_ms"] is None
    assert "3.1" in format_conclusion(res, "ru") and "ritm" in format_conclusion(res, "uz").lower()


def test_coarse_photo_escalates_in_mode_b_and_plain_signal_is_untouched():
    coarse = run_pipeline(_coarse(heart_rate_bpm=72), mode="B")
    assert coarse.report.mode == "escalate"
    plain = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    assert "photo_coarse" not in plain.findings.codes() and plain.measurements.intervals.qrs_ms is not None
