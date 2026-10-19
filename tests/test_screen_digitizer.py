"""Photos of a device's own screen (D120): lit panel, grey grid, N leads as rows, ~2 s."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.screen import (
    CHEST_6,
    LIMB_6,
    digitize_screen,
    is_device_screen,
    render_device_screen,
)
from cardiolens.layer1_ingestion.strip import digitize_auto_to_signal
from cardiolens.pipeline import run_pipeline


@pytest.fixture(scope="module")
def sig():
    return generate_synthetic_ecg(heart_rate_bpm=95, duration_s=10.0, seed=3)


def _r(a, b):
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1])


def test_chest_screen_is_read_and_ordered(sig, tmp_path):
    path = render_device_screen(sig, tmp_path / "s.png", leads=CHEST_6)
    import cv2

    assert is_device_screen(cv2.imread(path))
    traces = digitize_screen(path)
    assert list(traces) == CHEST_6                     # not limb: rows 1-3 break Einthoven
    n = int(2.5 * sig.sampling_rate_hz)
    rs = [_r(sig.lead(ln)[:n], traces[ln]) for ln in CHEST_6]
    assert min(rs) > 0.85, rs


def test_limb_screen_is_recognised_by_einthoven(sig, tmp_path):
    assert list(digitize_screen(render_device_screen(sig, tmp_path / "l.png", leads=LIMB_6))) == LIMB_6


def test_screen_photo_through_the_pipeline(sig, tmp_path):
    s = digitize_auto_to_signal(render_device_screen(sig, tmp_path / "p.png", leads=CHEST_6))
    assert set(s.provenance.lead_windows) == set(CHEST_6)
    r = run_pipeline(s, mode="A")
    assert r.measurements.heart_rate_bpm == pytest.approx(95, abs=4)
    assert "screen_photo_limited" in r.findings.codes()


def test_paper_page_is_not_a_screen(sig, tmp_path):
    import cv2

    from cardiolens.layer1_ingestion.digitize import render_ecg_page

    path = str(tmp_path / "page.png")
    render_ecg_page(sig, path)
    assert not is_device_screen(cv2.imread(path))


# --- D121: dark screens (bright trace on black), 6x2 displays, turned photos ---

@pytest.mark.parametrize("kw,layout", [
    (dict(dark=True), "screen 6x1"),
    (dict(dark=True, columns=2, seconds=5.0, row_pitch_mm=12, px_per_mm=6), "screen 6x2"),
    (dict(columns=2, seconds=5.0, row_pitch_mm=12, px_per_mm=6), "screen 6x2"),
])
def test_dark_and_two_column_screens(tmp_path, kw, layout):
    from cardiolens.layer1_ingestion.screen import digitize_screen_to_signal

    s = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0, seed=3)
    path = render_device_screen(s, tmp_path / "d.png", **kw)
    sig = digitize_screen_to_signal(path)
    calib = digitize_screen.last_calibration
    assert calib["layout"] == layout and calib["mode"] == ("dark" if kw.get("dark") else "light")
    if kw.get("columns") == 2:
        assert calib["columns"] == [LIMB_6, CHEST_6]
        assert sig.provenance.lead_windows["II"][1] <= sig.provenance.lead_windows["V1"][0] + 1e-6   # one after the other
    r = run_pipeline(sig, mode="A")
    assert r.measurements.heart_rate_bpm == pytest.approx(75, rel=0.1)


def test_turned_screen_photo_is_rotated_back(tmp_path):
    import cv2

    s = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0, seed=3)
    path = render_device_screen(s, tmp_path / "w.png", dark=True, columns=2, seconds=5.0, row_pitch_mm=12, px_per_mm=6)
    cv2.imwrite(str(tmp_path / "turned.png"), cv2.rotate(cv2.imread(path), cv2.ROTATE_90_COUNTERCLOCKWISE))
    digitize_screen(str(tmp_path / "turned.png"))
    assert digitize_screen.last_calibration["rotated_deg"] in (90, 270)
    assert digitize_screen.last_calibration["layout"] == "screen 6x2"


def test_white_box_on_a_dark_device_is_not_a_lit_panel(tmp_path):
    import cv2

    from cardiolens.layer1_ingestion.screen import screen_view

    s = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0, seed=3)
    view = screen_view(cv2.imread(render_device_screen(s, tmp_path / "b.png", dark=True)))
    assert view is not None and view[0] == "dark"


# --- D122: ~2-s windows (TeleECG preview) ---

@pytest.mark.parametrize("hr,seed", [(30, 7), (35, 4), (40, 20), (50, 12), (75, 3)])
def test_short_window_never_overstates_a_slow_rate(tmp_path, hr, seed):
    # One or no QRS in 1.8 s: the rate is not measured but bounded from above, and P/T
    # waves are not taken for beats (a 35/min record had been read as 105/min).
    from cardiolens.layer1_ingestion.screen import digitize_screen_to_signal

    s = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10.0, seed=seed)
    r = run_pipeline(digitize_screen_to_signal(render_device_screen(s, tmp_path / "w.png", seconds=1.8),
                                               lead_order=CHEST_6), mode="A")
    measured = r.measurements.heart_rate_bpm
    bound = next((float(f.evidence[0].value) for f in r.findings.findings if f.code == "rate_unmeasurable_short"), None)
    assert (measured is not None and measured == pytest.approx(hr, rel=0.1)) or (bound is not None and bound >= hr)


@pytest.mark.parametrize("kw,code", [(dict(heart_rate_bpm=70, bbb="rbbb", seed=6), "rbbb"),
                                     (dict(heart_rate_bpm=70, bbb="lbbb", seed=7), "lbbb")])
def test_two_beats_suffice_for_qrs_on_a_short_window(tmp_path, kw, code):
    from cardiolens.layer1_ingestion.screen import digitize_screen_to_signal

    s = generate_synthetic_ecg(duration_s=10.0, **kw)
    r = run_pipeline(digitize_screen_to_signal(render_device_screen(s, tmp_path / "b.png"), lead_order=CHEST_6),
                     mode="A")
    assert code in r.findings.codes()
