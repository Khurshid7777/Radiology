"""12-lead page digitisation (TZ §4, §17 "грязные фото"): render -> degrade -> recover."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import (
    LAYOUT_3X4,
    PANEL_SECONDS,
    degrade_image,
    digitize_page,
    digitize_page_to_signal,
    estimate_skew,
    render_ecg_page,
)
from cardiolens.pipeline import run_pipeline

DEGRADATIONS = {
    "rotation": {"rotation_deg": 3.0},
    "blur": {"blur_px": 2},
    "jpeg": {"jpeg_quality": 40},
    "shadow": {"shadow": 0.5},
    "salt": {"salt_noise": 0.005},
    "combo": {"rotation_deg": -2.0, "blur_px": 1, "jpeg_quality": 40, "shadow": 0.4, "salt_noise": 0.003},
}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1])


def _per_lead_r(sig, traces) -> list[float]:
    seg = int(PANEL_SECONDS * sig.sampling_rate_hz)
    return [
        _corr(sig.lead(lead)[col * seg:(col + 1) * seg], traces[lead])
        for row in LAYOUT_3X4 for col, lead in enumerate(row)
    ]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=10.0, seed=3)
    path = str(tmp_path_factory.mktemp("page") / "page.png")
    render_ecg_page(sig, path)
    return sig, path


def test_clean_page_roundtrip_all_leads(page):
    sig, path = page
    traces = digitize_page(path)
    assert set(traces) == {lead for row in LAYOUT_3X4 for lead in row} | {"II_rhythm"}
    rs = _per_lead_r(sig, traces)
    assert min(rs) > 0.85 and np.mean(rs) > 0.9
    assert digitize_page.last_coverage > 0.98
    # Rhythm strip covers the full 10 s of lead II.
    assert _corr(sig.lead("II"), traces["II_rhythm"]) > 0.9


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_degraded_page_roundtrip(page, tmp_path, name):
    sig, path = page
    dst = degrade_image(path, tmp_path / f"{name}.png", **DEGRADATIONS[name])
    traces = digitize_page(dst)
    rs = _per_lead_r(sig, traces)
    assert min(rs) > 0.8, (name, rs)
    assert np.mean(rs) > 0.88, (name, np.mean(rs))


def test_skew_is_estimated_and_undone(page, tmp_path):
    import cv2

    _sig, path = page
    dst = degrade_image(path, tmp_path / "rot.png", rotation_deg=3.0)
    # Sign follows the image y-axis (down); digitize_page feeds it straight back to warpAffine.
    assert abs(estimate_skew(cv2.imread(dst, cv2.IMREAD_GRAYSCALE))) == pytest.approx(3.0, abs=0.5)


def test_page_to_signal_windows_and_pipeline(page):
    sig, path = page
    dig = digitize_page_to_signal(path)
    assert dig.source == "image" and dig.provenance.digitized
    assert dig.provenance.confidence >= 0.95
    windows = dig.provenance.lead_windows
    assert windows["II"] == (0.0, 10.0)                # rhythm strip
    assert windows["V6"] == (7.5, 10.0)                # last column
    # Outside its window a panel lead is silent, inside it carries the trace.
    v6 = dig.lead("V6")
    fs = dig.sampling_rate_hz
    assert np.all(v6[: int(7.4 * fs)] == 0)
    assert np.abs(v6[int(7.5 * fs):]).max() > 200

    result = run_pipeline(dig, mode="A")
    assert result.measurements.heart_rate_bpm == pytest.approx(72, abs=3)
    assert result.measurements.axes.qrs_deg == pytest.approx(
        run_pipeline(sig, mode="A").measurements.axes.qrs_deg, abs=15)


def test_stemi_page_keeps_critical_finding(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=80, duration_s=10.0, st_elevation_uv=300,
                                 st_leads=["V2", "V3", "V4"], seed=5)
    path = str(tmp_path / "stemi.png")
    render_ecg_page(sig, path)
    dst = degrade_image(path, tmp_path / "stemi_photo.jpg", rotation_deg=2.0, blur_px=1,
                        jpeg_quality=40, shadow=0.3, salt_noise=0.002)
    result = run_pipeline(digitize_page_to_signal(dst), mode="A")
    assert "st_elevation" in result.findings.codes()


# ---- v3: grid-pitch scale, calibration pulse, perspective de-warp (D43) ----
def test_grid_pitch_is_estimated_from_the_image(tmp_path):
    import cv2

    from cardiolens.layer1_ingestion.digitize import estimate_grid_pitch

    sig = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=4)
    for ppm in (5, 6, 8):
        path = str(tmp_path / f"p{ppm}.png")
        render_ecg_page(sig, path, px_per_mm=ppm)
        est = estimate_grid_pitch(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        assert est is not None and abs(est - ppm) / ppm < 0.03, (ppm, est)


def test_calibration_pulse_gives_mm_per_mv(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=4)
    path = str(tmp_path / "pulse.png")
    render_ecg_page(sig, path, px_per_mm=6, mm_per_mv=10.0, calibration_pulse=True)
    traces = digitize_page(path, px_per_mm=None, mm_per_mv=None)
    cal = digitize_page.last_calibration
    assert cal["pulse_found"] and abs(cal["mm_per_mv"] - 10.0) < 0.5
    assert abs(cal["px_per_mm"] - 6.0) < 0.2
    rs = _per_lead_r(sig, traces)
    assert np.mean(rs) > 0.9 and min(rs) > 0.85
    # A page without a pulse must not mistake the first QRS for one.
    path2 = str(tmp_path / "nopulse.png")
    render_ecg_page(sig, path2, px_per_mm=6)
    digitize_page(path2, px_per_mm=None, mm_per_mv=None)
    assert digitize_page.last_calibration["pulse_found"] is False
    dig = digitize_page_to_signal(path, px_per_mm=None, mm_per_mv=None)
    assert any("calibration" in n for n in dig.provenance.notes)


def test_perspective_photo_is_dewarped(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10.0, seed=4)
    path = str(tmp_path / "page.png")
    render_ecg_page(sig, path, px_per_mm=6, calibration_pulse=True)
    photo = degrade_image(path, tmp_path / "photo.jpg", perspective=0.02, vignette=0.4, texture=0.3,
                          blur_px=1, jpeg_quality=45, rotation_deg=1.5, seed=1)
    with_dewarp = np.mean(_per_lead_r(sig, digitize_page(photo, px_per_mm=None, mm_per_mv=None, dewarp=True)))
    without = np.mean(_per_lead_r(sig, digitize_page(photo, px_per_mm=None, mm_per_mv=None, dewarp=False)))
    assert with_dewarp > 0.7 and with_dewarp > without + 0.15, (with_dewarp, without)


def test_strong_perspective_is_dewarped_by_the_page_quadrilateral(page, tmp_path):
    # 5 % corner displacement is beyond the vanishing-point method (D43 limit ~2 %);
    # the page quadrilateral (grid-textured hull, sides snapped to the outermost
    # grid lines) rectifies it when the whole page is in the frame (D48).
    sig, path = page
    dst = degrade_image(path, tmp_path / "persp5.png", perspective=0.05, seed=3)
    traces = digitize_page(dst)
    rs = _per_lead_r(sig, traces)
    assert np.mean(rs) > 0.85 and min(rs) > 0.75, rs


@pytest.mark.parametrize("hr,seed", [(38, 20), (40, 6), (64, 11)])
def test_clean_page_is_not_read_as_paced(tmp_path, hr, seed):
    # D114: at the pixel rate a QRS upstroke is a one-sample step that the raw-signal
    # spike detector took for a pacer spike (3 of 4 clean pages, HR 38 -> 53).
    sig = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10.0, seed=seed)
    path = str(tmp_path / "p.png")
    render_ecg_page(sig, path)
    r = run_pipeline(digitize_page_to_signal(path), mode="A")
    assert "paced_rhythm" not in r.findings.codes()
    assert r.measurements.heart_rate_bpm == pytest.approx(hr, abs=3)


def test_clean_page_passes_the_photo_gate(page):
    from cardiolens.layer2_preprocess.photo_quality import page_consistency

    sig, path = page
    c = page_consistency(digitize_page_to_signal(path))
    assert c["panel_lag_ms"] <= 10 and c["einthoven_r"] >= 0.85
    assert "photo_unreliable" not in run_pipeline(digitize_page_to_signal(path), mode="A").findings.codes()


def _af_photo(tmp_path, **degrade):
    sig = generate_synthetic_ecg(heart_rate_bpm=95, p_amp_scale=0.0, rr_jitter_ms=180, duration_s=10.0, seed=8)
    path = str(tmp_path / "p.png")
    render_ecg_page(sig, path)
    return run_pipeline(digitize_page_to_signal(degrade_image(path, tmp_path / "d.jpg", seed=1, **degrade)), mode="A")


def test_phone_photo_is_aligned_not_read_as_vt(tmp_path):
    # D114-D115: the rows of a phone-like photo sat 50-80 ms apart in time and the AF
    # record read as VT (QRS 70 -> 114 ms); shifting each panel onto the strip fixes it.
    r = _af_photo(tmp_path, perspective=0.02, rotation_deg=1.5, blur_px=1, jpeg_quality=60,
                  vignette=0.3, texture=0.2)
    codes = r.findings.codes()
    assert "photo_unreliable" not in codes and "atrial_fibrillation" in codes
    assert not r.safety.alerts and r.measurements.intervals.qrs_ms < 90


def test_wrecked_photo_withholds_alerts(tmp_path):
    r = _af_photo(tmp_path, perspective=0.08, rotation_deg=8.0, blur_px=5, jpeg_quality=15, vignette=0.8,
                  texture=0.8, shadow=0.7, salt_noise=0.02)
    assert r.findings.codes() == {"photo_unreliable"}
    assert not r.safety.alerts


def test_digital_record_has_no_photo_check():
    from cardiolens.layer2_preprocess.photo_quality import page_consistency

    assert page_consistency(generate_synthetic_ecg(heart_rate_bpm=70, seed=1)) is None
