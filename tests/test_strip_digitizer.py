"""12 x 1 strip digitiser (D72): rendered strips round-trip clean and as JPEG /
photo-like images, the layout switch, the companion de-warp and the gap bridges."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import (
    count_rows,
    degrade_image,
    digitize_auto_to_signal,
    digitize_strip_to_signal,
    generate_synthetic_ecg,
    render_ecg_page,
    render_ecg_strip,
)
from cardiolens.layer1_ingestion import digitize as D
from cardiolens.layer1_ingestion import strip as ST


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    root = tmp_path_factory.mktemp("strip")
    sig = generate_synthetic_ecg(heart_rate_bpm=72, seed=5)
    png = root / "strip.png"
    render_ecg_strip(sig, png, px_per_mm=6)
    return sig, png, root


def _corr(sig, got):
    fs_o, fs_g = sig.sampling_rate_hz, got.sampling_rate_hz
    out = {}
    for ln in sig.lead_names:
        a = sig.signal[sig.lead_names.index(ln)]
        b = got.signal[got.lead_names.index(ln)]
        n = min(int(len(a) / fs_o * fs_g), len(b))
        a_r = np.interp(np.arange(n) / fs_g, np.arange(len(a)) / fs_o, a)
        out[ln] = float(np.corrcoef(a_r, b[:n])[0, 1]) if np.std(b[:n]) > 0 else 0.0
    return out


def test_clean_strip_round_trip(strip):
    sig, png, _ = strip
    got = digitize_strip_to_signal(png)
    cal = ST.digitize_strip.last_calibration
    assert cal["layout"] == "12x1" and cal["pulses"] == 4 and abs(cal["mm_per_mv"] - 10.0) <= 0.5
    assert abs(cal["row_pitch_mm"] - ST.STRIP_ROW_PITCH_MM) <= 0.5 and abs(cal["px_per_mm"] - 6.0) <= 0.2
    r = _corr(sig, got)
    assert np.median(list(r.values())) >= 0.85 and min(r.values()) >= 0.7
    assert got.lead_names[1] == "II" and got.provenance.lead_windows["V6"][0] == 0.0
    assert got.provenance.confidence >= 0.95


def test_jpeg_and_photo_like_strip_still_read(strip):
    sig, png, root = strip
    jpg = root / "strip_q75.jpg"
    degrade_image(png, jpg, jpeg_quality=75, seed=3)
    r = _corr(sig, digitize_strip_to_signal(jpg))
    assert np.median(list(r.values())) >= 0.85                 # JPEG chroma noise must not cut the trace
    photo = root / "strip_photo.jpg"
    degrade_image(png, photo, perspective=0.02, rotation_deg=1.0, blur_px=1, jpeg_quality=75,
                  shadow=0.3, vignette=0.2, seed=3)
    r = _corr(sig, digitize_strip_to_signal(photo))
    assert np.median(list(r.values())) >= 0.75


def test_layout_switch_strip_vs_page(strip, tmp_path):
    sig, png, _ = strip
    assert count_rows(png) == 12
    page = tmp_path / "page.png"
    render_ecg_page(sig, page, px_per_mm=6, calibration_pulse=True)
    assert count_rows(page) == 4
    auto = digitize_auto_to_signal(png)
    assert any("12x1" in n for n in auto.provenance.notes)
    auto_page = digitize_auto_to_signal(page)
    assert any("3x4" in n for n in auto_page.provenance.notes)


def test_companion_warp_matches_the_dewarp(strip):
    import cv2

    _, _png, root = strip
    photo = root / "strip_photo.jpg"
    gray = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    out = D.dewarp_page(gray)
    again = D.apply_dewarp(gray, border=255)
    assert again.shape == out.shape
    h, w = out.shape
    centre = (slice(h // 6, 5 * h // 6), slice(w // 6, 5 * w // 6))
    blur = lambda im: cv2.GaussianBlur(im, (0, 0), 2)[centre].astype(int)
    assert np.abs(blur(again) - blur(out)).mean() < 2.0
    mask = gray < 100
    warped = D.apply_dewarp(mask, nearest=True)
    assert warped.dtype == bool and warped.shape == out.shape


def test_gap_bridges_are_dropped_but_tall_strokes_are_kept():
    pitch, half = 60.0, 90
    trace = np.full(200, float(half))
    trace[50:52] = half + pitch                         # two columns on the row below's flat baseline
    trace[120:123] = half - 0.9 * pitch                 # a tall R wave (same height, three columns)
    heights = np.full(200, 2.0)
    heights[120:123] = 40.0                             # ... drawn as a tall stroke
    out = ST._drop_short_jumps(trace, half, pitch, heights, stroke=2.0)
    assert np.isnan(out[50:52]).all() and np.isfinite(out[120:123]).all()
    plateau = np.full(200, float(half))
    plateau[10:20] = half + 1.05 * pitch                # ten columns riding the neighbour, flat ink
    out = ST._drop_foreign_plateaus(plateau, half, pitch, np.full(200, 2.0), 2.0)
    assert np.isnan(out[10:20]).all() and np.isfinite(out[:10]).all()


def test_telegram_analyse_reads_a_strip(tmp_path):
    """The bot's reader (D139): a 12x1 strip sent as a file (10 px/mm) is read by the scanner;
    the shared 6-px/mm fixture is below the scanner's 8-px/mm floor and is refused."""
    from cardiolens.telegram.analyze import analyse_photo, format_card
    from cardiolens.telegram.store import Record

    png = render_ecg_strip(generate_synthetic_ecg(heart_rate_bpm=70, seed=2), tmp_path / "s.png", px_per_mm=10)
    result, error = analyse_photo(png)
    assert error is None and "sinus_rhythm" in result.findings.codes()
    card = format_card(Record(id="1_1", chat_id=1, msg_id=1), result)
    assert "раскладка 12x1 (сканер)" in card
