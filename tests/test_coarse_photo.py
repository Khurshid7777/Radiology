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
    # ST of 2.5 mm survives the raised thresholds, said to be approximate; its alert is unverified.
    st = next(f for f in res.findings.findings if f.code == "st_elevation")
    assert "ориентировочно" in st.label_ru and st.confidence < 0.8
    assert not any(f.group in ("interval", "hypertrophy") for f in res.findings.findings)
    alert = next(a for a in res.safety.alerts if a.code == "st_elevation")
    assert alert.unverified
    assert res.measurements.heart_rate_bpm is not None and abs(res.measurements.heart_rate_bpm - 72) < 4
    iv = res.measurements.intervals
    assert iv.pr_ms is None and iv.qrs_ms is None and iv.qt_ms is None and iv.qtc_bazett_ms is None
    d = result_to_dict(res)
    assert d["measurements"]["intervals"]["qrs_ms"] is None
    assert "3.1" in format_conclusion(res, "ru") and "ritm" in format_conclusion(res, "uz").lower()


def test_coarse_photo_escalates_in_mode_b_and_plain_signal_is_untouched():
    coarse = run_pipeline(_coarse(heart_rate_bpm=72), mode="B")
    assert coarse.report.mode == "escalate"
    plain = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    assert "photo_coarse" not in plain.findings.codes() and plain.measurements.intervals.qrs_ms is not None


def test_coarse_scan_is_not_handed_to_the_readers(monkeypatch, tmp_path):
    """D170: a coarse scanner read stays with the scanner even when D144 would doubt its scale."""
    import cardiolens.telegram.analyze as A

    class Cal:
        confidence = 0.0

    class Res:
        status, warnings, calibration, artifacts = "needs_confirmation", ["LOW_RESOLUTION_COARSE", "SPEED_UNCERTAIN", "RHYTHM_INCONSISTENT"], Cal(), {}

    sig = _coarse(heart_rate_bpm=72)
    import ecgscan.contract.cardiolens_adapter as AD
    import ecgscan.pipeline as P
    monkeypatch.setattr(P, "scan", lambda image, **kw: Res())
    monkeypatch.setattr(AD, "to_signal_representation", lambda r: sig)
    monkeypatch.setattr(A, "digitize_auto_to_signal", lambda p: (_ for _ in ()).throw(AssertionError("readers must not run")))
    photo = tmp_path / "p.png"; photo.write_bytes(b"\x89PNG")
    assert A.digitize_photo(photo) is sig


def test_coarse_photo_small_st_is_not_called():
    """A 1-mm shift is three pixels at 3 px/mm: below the raised thresholds, not reported."""
    res = run_pipeline(_coarse(heart_rate_bpm=72, st_elevation_uv=120, st_leads=["II", "III", "aVF"]), mode="A")
    assert not any(f.group == "ischemia" for f in res.findings.findings)
    plain = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72, st_elevation_uv=120, st_leads=["II", "III", "aVF"]), mode="A")
    assert "st_elevation" in plain.findings.codes()        # the same ECG at full scale is called


def test_a_page_photographed_sideways_is_refused_with_the_turn_advice(tmp_path):
    """D171: the scanner refuses the layout of a tall photo; the readers used to read it as a
    12x1 strip (time running down the page). Now it is refused with the advice to turn it."""
    import cv2
    import pytest

    from cardiolens.layer1_ingestion.digitize import render_ecg_page
    from cardiolens.telegram.analyze import digitize_photo

    png = tmp_path / "p.png"
    render_ecg_page(generate_synthetic_ecg(heart_rate_bpm=72, seed=5), png, px_per_mm=10, calibration_pulse=True)
    cv2.imwrite(str(tmp_path / "rot.png"), cv2.rotate(cv2.imread(str(png)), cv2.ROTATE_90_CLOCKWISE))
    with pytest.raises(ValueError, match="Поверните"):
        digitize_photo(tmp_path / "rot.png")
