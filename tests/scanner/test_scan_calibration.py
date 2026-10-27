"""M2: grid, pulse, header OCR, calibration rules and statuses (TZ §5.3-5.4, §14 M2).

The TZ asks for tests at 25 and 50 mm/s and 5, 10, 20 mm/mV; a wrong speed must
never come back as ``ok`` (it halves or doubles the heart rate).
"""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.calibration.header_ocr import ocr_available, parse_settings
from ecgscan.calibration.pulse import find_pulses, ink_mask, summarize_pulses
from ecgscan.calibration.resolve import resolve
from ecgscan.geometry.grid import estimate_grid
from ecgscan.layout import get_layout
from ecgscan.pipeline import ScanDebug, ScanHints, pipeline_config, scan

cv2 = pytest.importorskip("cv2")
needs_ocr = pytest.mark.skipif(not ocr_available()[0], reason="OCR engine (rapidocr-onnxruntime) not installed")


def _page(synth_mv, *, ppm=10.0, speed=25, gain=10, header=True, layout="3x4+rhythm", grid="red"):
    img, _ = render_page(synth_mv, 500.0, get_layout(layout),
                         RenderSpec(px_per_mm=ppm, speed_mm_s=speed, gain_mm_mv=gain, header=header, grid=grid))
    return img


@pytest.mark.parametrize("ppm,grid", [(8.0, "red"), (9.5, "pink"), (12.0, "green"), (15.0, "gray")])
def test_grid_pitch_any_scale_and_colour(synth_mv, ppm, grid):
    g = estimate_grid(_page(synth_mv, ppm=ppm, grid=grid))
    assert g.px_per_mm == pytest.approx(ppm, rel=0.01) and g.confidence >= 0.5


def test_no_grid_is_not_invented():
    rng = np.random.default_rng(0)
    noise = rng.integers(200, 256, (800, 1200, 3), dtype=np.uint8)
    g = estimate_grid(noise)
    assert g.px_per_mm is None or g.confidence < 0.5


@pytest.mark.parametrize("gain", [5, 10, 20])
@pytest.mark.parametrize("speed", [25, 50])
def test_pulse_height_is_the_gain(synth_mv, speed, gain):
    img = _page(synth_mv, speed=speed, gain=gain)
    s = summarize_pulses(find_pulses(ink_mask(img, 10.0), 10.0))
    assert s is not None and s["n_agree"] >= 3
    assert s["height_mm"] == pytest.approx(gain, rel=0.06)            # incl. 20 mm reaching the next row
    area = get_layout("3x4+rhythm").pulse.area_mm
    assert s["width_mm"] == pytest.approx(min(0.2 * speed, area - 2.0), rel=0.15)   # 200 ms, capped by its area


def test_settings_parser():
    sp, gn = parse_settings(["25mm/s10mm/mV0.05-150Hz", "Name:TEST", "50 mm/s  20 mm/mV", "12,5mm/s"])
    assert sp == [25.0, 50.0, 12.5] and gn == [10.0, 20.0]
    assert parse_settings(["0.05-150Hz 50Hz"]) == ([], [])
    # RapidOCR 1.4 splits a header line into boxes
    assert parse_settings(["Name:", "TEST", "25", "mm/s", "10", "mm/mV", "0.05-150"]) == ([25.0], [10.0])


CAL = pipeline_config()["calibration"]


def _resolve(**kw):
    base = dict(px_per_mm=10.0, grid_residual_mm=0.1, pulse=None, ocr_speeds=[], ocr_gains=[], cfg=CAL)
    base.update(kw)
    return resolve(**base)


def test_resolve_ocr_and_pulse_agree():
    d = _resolve(pulse={"height_mm": 20.3, "width_mm": 10.1, "n_agree": 4}, ocr_speeds=[50.0], ocr_gains=[20.0])
    assert d.established and d.calibration.paper_speed_mm_s == 50 and d.calibration.gain_mm_mv == 20
    assert d.calibration.speed_source == "header_ocr" and d.calibration.gain_source == "calibration_pulse"
    assert d.warnings == []


def test_pulse_width_alone_never_establishes_speed():
    d = _resolve(pulse={"height_mm": 10.0, "width_mm": 10.0, "n_agree": 4})
    assert not d.established and d.calibration.paper_speed_mm_s is None
    assert "SPEED_UNCERTAIN" in d.warnings and d.calibration.gain_mm_mv == 10
    assert [c.value for c in d.calibration.speed_candidates] == [50.0] and d.speed_for_traces == 50.0


def test_disagreements_need_confirmation():
    d = _resolve(pulse={"height_mm": 10.0, "width_mm": 5.0, "n_agree": 4}, ocr_speeds=[50.0], ocr_gains=[20.0])
    assert not d.established
    assert {"SPEED_SOURCES_DISAGREE", "GAIN_SOURCES_DISAGREE"} <= set(d.warnings)


def test_ocr_gain_without_pulse_needs_confirmation_and_user_decides():
    d = _resolve(ocr_speeds=[25.0], ocr_gains=[10.0])
    assert d.calibration.paper_speed_mm_s == 25 and d.calibration.gain_mm_mv is None and "GAIN_UNCERTAIN" in d.warnings
    u = _resolve(ocr_speeds=[25.0], ocr_gains=[10.0], user_gain=20.0, user_speed=50.0)
    assert u.established and u.calibration.gain_mm_mv == 20 and u.calibration.speed_source == "user"


def test_nothing_known_is_not_defaulted():
    d = _resolve()
    assert d.calibration.paper_speed_mm_s is None and d.calibration.gain_mm_mv is None
    assert {"SPEED_UNCERTAIN", "GAIN_UNCERTAIN"} <= set(d.warnings) and d.calibration.confidence == 0.0


def _encode(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


@needs_ocr
@pytest.mark.parametrize("speed,gain", [(25, 10), (50, 20), (25, 5)])
def test_scan_reads_speed_and_gain(synth_mv, speed, gain):
    res = scan(_encode(_page(synth_mv, ppm=11.0, speed=speed, gain=gain)))
    assert res.status == "ok", (res.warnings, res.reject_reasons)
    assert res.calibration.paper_speed_mm_s == speed and res.calibration.gain_mm_mv == gain
    assert res.layout.name == "3x4+rhythm" and len(res.leads) == 12 and res.rhythm[0].lead == "II"
    assert all(abs(v) < 3.0 for t in res.leads.values() for v in t.samples_mv if v is not None)
    assert res.calibration.px_per_mm_x == pytest.approx(11.0, rel=0.01)


@needs_ocr
def test_50_mm_s_without_header_is_never_ok(synth_mv):
    """TZ §5.4: 50 read as 25 halves the rate; without OCR the speed is not established."""
    res = scan(_encode(_page(synth_mv, ppm=11.0, speed=50, gain=10, header=False)))
    assert res.status == "needs_confirmation" and res.calibration.paper_speed_mm_s is None
    assert "SPEED_UNCERTAIN" in res.warnings
    # The doctor's confirmation establishes it.
    ok = scan(_encode(_page(synth_mv, ppm=11.0, speed=50, gain=10, header=False)),
              hints=ScanHints(paper_speed_mm_s=50))
    assert ok.status == "ok" and ok.calibration.speed_source == "user"


def test_low_resolution_and_no_grid_are_rejected(synth_mv):
    # D170 / SD-074: a small image below the floor is read coarse, never "ok"; a large one
    # (a phone original) below the floor is still refused.
    small = scan(_encode(_page(synth_mv, ppm=6.0)))
    assert small.status == "needs_confirmation" and "LOW_RESOLUTION_COARSE" in small.warnings
    big = cv2.copyMakeBorder(_page(synth_mv, ppm=6.0), 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=(120, 110, 100))
    low = scan(_encode(big))
    assert low.status == "rejected" and low.reject_reasons == ["LOW_RESOLUTION"] and not low.leads
    blank = scan(_encode(np.full((900, 1400, 3), 250, np.uint8)))
    assert blank.status == "rejected" and blank.reject_reasons == ["GRID_NOT_FOUND"]


def test_faded_print_is_read_for_confirmation_and_a_blank_one_rejected(synth_mv):
    """SD-064: faded thermal paper (contrast ~0.65) is read, flagged FADED_PAPER and sent to
    the doctor; a trace too pale to see (contrast < 0.15) is still refused with the reason."""
    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"),
                         RenderSpec(px_per_mm=10.0, trace_gray=150, grid_fade=0.4))
    res = scan(_encode(img), debug=ScanDebug())
    assert res.status == "needs_confirmation" and "FADED_PAPER" in res.warnings and res.leads
    ghost, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"),
                           RenderSpec(px_per_mm=10.0, trace_gray=235, grid_fade=0.4))
    res = scan(_encode(ghost), debug=ScanDebug())
    assert res.status == "rejected" and res.reject_reasons == ["LOW_CONTRAST"]


def test_original_is_hashed_as_received(synth_mv):
    import hashlib

    data = _encode(_page(synth_mv, ppm=6.0))
    res = scan(data)
    assert res.source.original_sha256 == hashlib.sha256(data).hexdigest()
    assert res.source.original_bytes == len(data) and res.source.mime_type == "image/png"


def test_a_piece_of_a_pulse_is_not_a_second_pulse():
    """M6: the upper half of a pulse, its edges seen again inside it, read as a second pulse
    5 mm above the row and became a fifth row (row_pitch_38-001, LAYOUT_UNKNOWN)."""
    m = np.zeros((300, 200), bool)
    m[100:205, 50:52] = m[100:205, 100:102] = True          # 10.5 mm x 5 mm pulse at 10 px/mm
    m[100:102, 50:102] = True
    m[100:150, 56:58] = m[100:150, 94:96] = True             # a half-height piece inside it
    assert [p.bottom for p in find_pulses(m, 10.0)] == [205]


def test_a_sheared_page_is_de_sheared_not_only_de_rotated(synth_mv):
    """M6: rows tilted 2.7 deg, columns -1.5 deg (an affine photo); one rotation left 1.2 deg
    of shear, ~20 ms of time offset per row, and the page was misread (photo_clean-002)."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import estimate_skew
    from ecgscan.geometry.rectify import rectify

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    h, v = np.radians(2.7), np.radians(-1.5)
    a = np.array([[np.cos(h), np.sin(v)], [np.sin(h), np.cos(v)]])
    pad = 150
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    c = np.array(canvas.shape[1::-1]) / 2
    sheared = cv2.warpAffine(canvas, np.hstack([a, (c - a @ c)[:, None]]), canvas.shape[1::-1],
                             flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    for shear_min, ok in ((None, False), (0.3, True)):
        r = rectify(sheared, working_px_per_mm=10.0, shear_min_deg=shear_min)
        rows = estimate_skew(r.gray)
        cols = estimate_skew(np.ascontiguousarray(r.gray.T))
        assert (abs(rows + cols) < 0.3) is ok, (rows, cols)


def test_line_angles_reads_rows_and_columns_in_one_pass():
    import cv2

    from ecgscan.geometry.rectify import line_angles

    g = np.full((900, 1200), 255, np.uint8)
    for y in range(100, 800, 50):                            # rows tilted +2 deg
        cv2.line(g, (50, y), (1150, y + int(round(1100 * np.tan(np.radians(2))))), 0, 1)
    for x in range(100, 1100, 50):                           # columns tilted -1 deg (dx/dy)
        cv2.line(g, (x, 20), (x + int(round(860 * np.tan(np.radians(-1)))), 880), 0, 1)
    rows, cols = line_angles(g)
    assert rows == pytest.approx(2.0, abs=0.3) and cols == pytest.approx(-1.0, abs=0.3)


def test_a_piece_of_a_page_is_incomplete_not_broken(synth_mv):
    """SD-067: the left 40 % of a 3x4 + rhythm page (a strip torn or photographed in part)
    is refused as PAGE_INCOMPLETE — a layout matter — not as TRACE_BROKEN (a final refusal)."""
    img, gt = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
    rhythm = next(t for t in gt.traces if t.kind == "rhythm")
    cut = int(rhythm.x0_px + 0.4 * 10 * 25 * 10)
    res = scan(cv2.imencode(".png", img[:, :cut])[1].tobytes(), hints=ScanHints(layout="3x4+rhythm"))
    assert res.reject_reasons == ["PAGE_INCOMPLETE"]


def test_a_grid_re_read_far_from_the_predicted_scale_is_not_used(synth_mv, monkeypatch):
    """SD-068: after the perspective fit the grid of a faint page was re-read at 17.9 px/mm
    (confidence 0.03); the page was squeezed 1.8x in time, the pulse read as 5 mm/mV and the
    heart rate came out 100 instead of 55. The axis now keeps the predicted scale."""
    import ecgscan.geometry.rectify as R
    from ecgscan.geometry.grid import GridEstimate

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0))
    h, w = img.shape[:2]
    quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    tilt = np.float32([[60, 40], [w - 20, 0], [w, h], [0, h - 50]])
    photo = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, tilt), (w, h), borderValue=(255, 255, 255))
    raw = estimate_grid(photo)
    real = R.estimate_grid
    calls = []

    def harmonic(bgr):
        g = real(bgr)
        calls.append(g)
        return GridEstimate(17.9, g.px_per_mm_y, 0.3, 0.4, "gray", 0.03) if len(calls) == 1 else g

    monkeypatch.setattr(R, "estimate_grid", harmonic)
    rect = R.rectify(photo, working_px_per_mm=10.0, raw_grid=raw, max_upscale=1.25)
    assert rect.method != "affine" and calls
    assert rect.grid.px_per_mm_x == pytest.approx(raw.px_per_mm * min(1.25, 10.0 / raw.px_per_mm), rel=0.01)
    assert rect.bgr.shape[1] == pytest.approx(img.shape[1] * 10.0 / 10.0, rel=0.1)    # not squeezed 1.8x
