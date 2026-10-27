"""D170 / SD-074: a messenger copy (~3-4 px/mm, the 1-mm lines lost to JPEG) is read in the
coarse mode — 5-mm grid, brought up to 10 px/mm, never "ok" — and nothing else changes."""

import numpy as np
import pytest

from ecgscan.bench.degrade import DegradeSpec, degrade
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.geometry.grid import GridEstimate, estimate_coarse_grid
from ecgscan.layout import get_layout
from ecgscan.pipeline import _coarse_grid, pipeline_config, scan

cv2 = pytest.importorskip("cv2")


def _messenger(mv, layout="3x4+rhythm", background=0.03, seed=3):
    """A photo of the page sent through a messenger: 1280 px on the long side, JPEG 80."""
    page, _ = render_page(mv, 500.0, get_layout(layout), RenderSpec(px_per_mm=8.0))
    spec = DegradeSpec(background=background, rotation_deg=1.5, vignette=0.1, noise=2, blur_px=0.6,
                       max_side_px=1280, jpeg_quality=80)
    img, info = degrade(page, spec, seed=seed, px_per_mm=8.0)
    return img, 8.0 * info["centre_scale"]


@pytest.mark.parametrize(("layout", "background"), [("3x4+rhythm", 0.03), ("6x2", 0.22)])
def test_a_messenger_copy_is_read_coarse_and_never_ok(synth_mv, layout, background):
    img, true_ppm = _messenger(synth_mv, layout, background)
    assert max(img.shape[:2]) == 1280 and true_ppm < 5
    g = estimate_coarse_grid(img)
    assert g is not None and g.px_per_mm == pytest.approx(true_ppm, rel=0.04)
    res = scan(cv2.imencode(".png", img)[1].tobytes())
    assert res.status == "needs_confirmation" and "LOW_RESOLUTION_COARSE" in res.warnings
    assert res.quality.px_per_mm == pytest.approx(true_ppm, rel=0.04)
    assert res.layout.name == layout
    hr_true, hr = _hr(synth_mv["II"]), _hr(res.leads["II"].array())
    assert hr == pytest.approx(hr_true, abs=5)


def _hr(x, fs=500.0):
    from scipy.signal import find_peaks

    x = np.nan_to_num(np.asarray(x, float) - np.nanmedian(x))
    p, _ = find_peaks(x, height=0.5 * np.nanmax(x), distance=int(0.3 * fs))
    return 60.0 / (np.median(np.diff(p)) / fs)


def test_the_coarse_mode_carries_its_note_to_cardiolens(synth_mv):
    from ecgscan.contract.cardiolens_adapter import to_signal_representation

    img, _ = _messenger(synth_mv)
    res = scan(cv2.imencode(".png", img)[1].tobytes())
    notes = to_signal_representation(res).provenance.notes
    assert any(n.startswith("coarse: px_per_mm=") for n in notes)


def test_large_or_sharp_images_never_take_the_coarse_path(synth_mv):
    cfg = pipeline_config()
    big = np.full((2600, 3600, 3), 250, np.uint8)
    fine = GridEstimate(12.0, 12.0, 3.0, 3.0, "chroma", 1.0)
    assert _coarse_grid(big, GridEstimate(None, None, 0, 0, "chroma", 0.0), cfg) is None    # a phone original
    small = np.full((900, 1280, 3), 250, np.uint8)
    assert _coarse_grid(small, GridEstimate(8.0, 8.0, 3.0, 3.0, "chroma", 1.0), cfg) is None  # 160 mm: plausible
    # 12 px/mm on 1280 px would be a 107-mm page: a misread, so the 5-mm grid is asked (none here).
    assert _coarse_grid(small, fine, cfg) is None
    off = {**cfg, "coarse": {**cfg["coarse"], "enabled": False}}
    img, _ = _messenger(synth_mv)
    assert _coarse_grid(img, GridEstimate(None, None, 0, 0, "gray", 0.0), off) is None
    assert _coarse_grid(img, GridEstimate(None, None, 0, 0, "gray", 0.0), cfg) is not None


def test_no_5mm_grid_still_refuses():
    blank = np.full((900, 1280, 3), 245, np.uint8)
    cv2.putText(blank, "not an ECG", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 5)
    assert estimate_coarse_grid(blank) is None
    res = scan(cv2.imencode(".png", blank)[1].tobytes())
    assert res.status == "rejected" and "LOW_RESOLUTION_COARSE" not in res.warnings


def test_coarse_slices_stay_out_of_the_standard_suite():
    from ecgscan.bench.cases import bench_config, suite_slices

    cfg = bench_config()
    _, std = suite_slices(cfg, "standard")
    _, coarse = suite_slices(cfg, "coarse")
    assert not [s for s in std if s.startswith("coarse_")] and "photo_clean" in std
    assert coarse == ["coarse_3x4_3ppm", "coarse_3x4_4ppm", "coarse_6x2_3ppm", "coarse_6x2_4ppm"]


def test_the_warning_says_how_coarse_in_both_languages():
    from ecgscan.i18n import t
    from ecgscan_service.router import _fmt

    res = {"quality": {"px_per_mm": 3.14}}
    assert "≈3,1 " in t("warning.LOW_RESOLUTION_COARSE", "ru", **_fmt(res, "ru"))
    assert "≈3.1 " in t("warning.LOW_RESOLUTION_COARSE", "uz", **_fmt(res, "uz"))
    assert t("warning.LOW_RESOLUTION_MARGINAL", "ru", **_fmt({}, "ru"))     # other texts ignore the placeholder


def test_a_sheet_lost_in_a_busy_frame_is_found_by_tiles(synth_mv):
    """A strip on a carpet or a lap: when the whole frame's comb is diluted the tiles vote;
    a frame with no grid (a device screen, a non-ECG picture) gives nothing either way."""
    rng = np.random.default_rng(1)
    page, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=5.0))
    bg = cv2.GaussianBlur((rng.random((960, 1280, 3)) * 255).astype(np.uint8), (0, 0), 2.5)
    img = bg.copy()
    strip = page[300:700, :1220]
    img[420:420 + strip.shape[0], 60:60 + strip.shape[1]] = strip
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)
    g = estimate_coarse_grid(img, min_score=99.0)                 # the whole-frame step refused: tiles only
    assert g is not None and g.channel == "tiles:5mm" and g.px_per_mm == pytest.approx(5.0, rel=0.03)
    assert estimate_coarse_grid(bg) is None and estimate_coarse_grid(bg, min_score=99.0) is None

