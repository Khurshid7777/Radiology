"""M3: segmentation, panels, vectorisation, method fusion, limb-lead checks, artifacts (TZ §5.5-5.9, §14 M3)."""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.contract import Trace
from ecgscan.layout import get_layout
from ecgscan.layout.panels import PageFrame, fill_gaps, panels, pick_rows
from ecgscan.pipeline import ScanDebug, ScanHints, scan
from ecgscan.segmentation import ClassicSegmenter, OnnxSegmenter, make_segmenter, paper_mask, thin
from ecgscan.validate import limb_consistency
from ecgscan.vectorize import fuse
from ecgscan.vectorize.trace import resample_nan, trace_panel

cv2 = pytest.importorskip("cv2")
FS = 500.0
HINTS = ScanHints(paper_speed_mm_s=25, gain_mm_mv=10)


def _page(synth_mv, layout="3x4+rhythm", **kw):
    spec = dict(px_per_mm=10.0, speed_mm_s=25, gain_mm_mv=10)
    spec.update(kw)
    img, _ = render_page(synth_mv, FS, get_layout(layout), RenderSpec(**spec))
    return img


def _encode(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


# --- segmentation -------------------------------------------------------------------------

def test_classic_segmenter_keeps_trace_drops_grid(synth_mv):
    img = _page(synth_mv)
    m = ClassicSegmenter().segment(img, 10.0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert m.ink.dtype == bool and m.ink.shape == gray.shape
    assert 0.002 < m.ink.mean() < 0.05                     # strokes, not the grid
    assert gray[m.ink].mean() < 0.5 * gray[~m.ink].mean()


def test_paper_mask_excludes_a_dark_desk(synth_mv):
    img = _page(synth_mv)
    h, w = img.shape[:2]
    canvas = np.full((h + 400, w + 400, 3), 60, np.uint8)  # a plain desk around the page
    canvas[200:200 + h, 200:200 + w] = img
    pm = paper_mask(canvas, 10.0)
    assert pm[250:150 + h, 250:150 + w].mean() > 0.95
    assert pm[:150].mean() < 0.05 and pm[:, :150].mean() < 0.05


def test_onnx_segmenter_without_model_is_unavailable_not_silent():
    ok, why = OnnxSegmenter(None).available()
    assert not ok and "no ONNX segmentation model" in why
    with pytest.raises(RuntimeError):
        make_segmenter({"kind": "onnx", "onnx_model": None})


def _committed_model() -> tuple[str, str]:
    import json

    from ecgscan.config import repo_root

    card = json.loads((repo_root() / "models/segmenter/model_card.json").read_text())
    return "models/segmenter/trace_unet.onnx", card["sha256"]


def test_onnx_segmenter_refuses_a_model_that_is_not_its_card():
    path, _ = _committed_model()
    ok, why = OnnxSegmenter(path, sha256="0" * 64).available()
    assert not ok and "sha256" in why


def test_onnx_segmenter_finds_the_strokes_the_classic_one_finds(synth_mv):
    pytest.importorskip("onnxruntime")
    path, sha = _committed_model()
    img = _page(synth_mv, header=False)
    seg = OnnxSegmenter(path, sha256=sha, tile=512)               # several tiles and seams on one page
    assert seg.available() == (True, None)
    m = seg.segment(img, 10.0)
    ref = ClassicSegmenter().segment(img, 10.0).ink
    assert m.ink.shape == ref.shape and m.darkness.dtype == np.float32
    inter = (m.ink & ref).sum()
    assert inter / ref.sum() > 0.85 and inter / m.ink.sum() > 0.7  # the same strokes, not the grid
    whole = OnnxSegmenter(path, sha256=sha, tile=4096).probability(img)
    assert np.abs(whole - seg.probability(img)).max() < 0.05      # tiling does not show


def test_thinning_gives_one_pixel_line():
    m = np.zeros((20, 60), bool)
    m[8:12, 5:55] = True
    sk = thin(m)
    assert sk[:, 10:50].sum(axis=0).max() == 1 and sk[:, 10:50].any(axis=0).all()


# --- panels ----------------------------------------------------------------------------------

def test_pick_rows_drops_a_stray_baseline():
    assert pick_rows([100, 400, 430, 700, 1000, 1300], 5, 300.0) == [100, 400, 700, 1000, 1300]


def test_fill_gaps_inserts_a_missed_row():
    assert fill_gaps([100, 400, 1000], 300.0) == [100, 400, 700, 1000]


def test_panels_follow_calibrated_speed_not_ink_extent():
    spec = get_layout("3x4+rhythm")
    frame = PageFrame([100.0, 400.0, 700.0, 1000.0], x0=50.0, ink_x1=None, implied_speed_mm_s=None, pitch_px=300.0)
    ps = panels(spec, frame, 10.0, 25.0)
    assert len(ps) == 13
    v1 = next(p for p in ps if p.lead == "V1")
    assert (v1.x0, v1.x1) == (50 + 2 * 625, 50 + 3 * 625)
    rhythm = [p for p in ps if p.kind == "rhythm"]
    assert rhythm[0].x1 - rhythm[0].x0 == 2500 and rhythm[0].baseline == 1000.0


# --- vectorisation -----------------------------------------------------------------------------

def _draw(y: np.ndarray, h: int = 200, thick: int = 3) -> np.ndarray:
    img = np.zeros((h, len(y)), np.uint8)
    pts = np.stack([np.arange(len(y)), np.round(y)], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, 255, thick)
    return img > 0


def test_tracer_follows_a_spike_and_keeps_its_peak():
    x = np.arange(400)
    y = 100 - 70 * np.exp(-0.5 * ((x - 200) / 4.0) ** 2)    # a narrow R wave, 70 px tall
    ink = _draw(y)
    pt = trace_panel(ink, ink.astype(np.float32), 0, 400, 0, 200, 100.0, stroke_px=3.0, pitch_px=150.0,
                     skeleton=thin(ink))
    assert np.isfinite(pt.y).mean() > 0.97
    assert np.nanmin(pt.y) == pytest.approx(y.min(), abs=2.5)       # the peak is not flattened
    ok = np.isfinite(pt.y)
    assert np.median(np.abs(pt.y[ok] - y[ok])) < 1.0


def test_narrow_spike_drawn_with_a_thick_pen_keeps_its_tip():
    """A pediatric R (~3 columns) with a 7-px stroke is a solid wedge; its skeleton breaks
    up and the path ran along the base (0.87 mV read as 0.06 mV)."""
    y = np.full(300, 150.0)
    y[148:153] = [120.0, 90.0, 70.0, 90.0, 120.0]
    ink = _draw(y, thick=7)
    pt = trace_panel(ink, ink.astype(np.float32), 0, 300, 0, 200, 150.0, stroke_px=7.0, pitch_px=150.0,
                     skeleton=thin(ink))
    assert np.nanmin(pt.y) == pytest.approx(70.0, abs=4.0)
    assert np.nanmedian(np.abs(pt.y[:140] - 150.0)) < 1.0


def test_missing_ink_stays_missing():
    y = np.full(300, 100.0)
    ink = _draw(y)
    ink[:, 120:180] = False
    pt = trace_panel(ink, ink.astype(np.float32), 0, 300, 0, 200, 100.0, stroke_px=3.0, pitch_px=150.0)
    assert np.isnan(pt.y[125:175]).all() and np.isfinite(pt.y[:110]).all()


def test_tracer_prefers_own_row_over_a_neighbours_trace():
    ink = _draw(np.full(300, 100.0)) | _draw(np.full(300, 140.0))   # a neighbour row's ink in the band
    pt = trace_panel(ink, ink.astype(np.float32), 0, 300, 40, 200, 100.0, stroke_px=3.0, pitch_px=150.0,
                     other_baselines=[140.0])
    assert np.nanmedian(np.abs(pt.y - 100.0)) < 1.0


def test_resample_never_bridges_gaps():
    v = np.ones(100)
    v[40:60] = np.nan
    out = resample_nan(v, 250.0, 500.0, 0.4)
    assert len(out) == 200 and np.isnan(out[82:118]).all() and np.isfinite(out[:78]).all()


# --- fusion ---------------------------------------------------------------------------------------

def test_fuse_takes_detail_at_a_short_apex():
    path = np.full(100, 300.0)
    detail = path.copy()
    detail[48:52] = [290.0, 250.0, 252.0, 290.0]            # the tip the path stopped short of
    out = fuse(path, detail, 100.0, far_pitches=1.0, margin=2)
    assert out[49] == 250.0


def test_fuse_rejects_a_long_excursion_and_another_rows_ink():
    path = np.full(200, 300.0)
    detail = path.copy()
    detail[50:90] = 330.0                                    # D47 riding something else for 40 columns
    detail[150:155] = 398.0                                  # on the next row's baseline
    out = fuse(path, detail, 100.0, margin=2, baseline=300.0, other_baselines=[400.0])
    assert (out[50:90] == 300.0).all() and (out[150:155] == 300.0).all()


def test_fuse_never_fills_a_path_gap_at_the_panel_start_from_detail():
    """Photo bug (M3): at a row start the DP path was NaN and D47 had jumped into the header."""
    path = np.full(200, 600.0)
    path[:30] = np.nan
    detail = np.full(200, 600.0)
    detail[:30] = 300.0
    out = fuse(path, detail, 300.0, margin=2)
    assert np.isnan(out[:30]).all()


def test_fuse_fills_a_short_bounded_gap():
    path = np.full(200, 600.0)
    path[100:104] = np.nan                                   # a steep edge the path skipped
    detail = np.full(200, 600.0)
    detail[100:104] = [580.0, 540.0, 560.0, 590.0]
    out = fuse(path, detail, 300.0, margin=0)
    assert np.array_equal(out[100:104], detail[100:104])


# --- limb-lead consistency ----------------------------------------------------------------------

def _limb_traces(mv: dict, *, swap: bool = False) -> list[tuple[str, Trace]]:
    n = int(2.5 * FS)
    out = []
    for ld in ("I", "II", "III", "aVR", "aVL", "aVF"):
        src = {"I": "II", "II": "I"}.get(ld, ld) if swap else ld
        out.append((ld, Trace.from_array(mv[src][:n], start_s=0.0, fs=FS, quality=1.0)))
    return out


def test_limb_identities_hold_for_physiological_leads(synth_mv):
    c = limb_consistency(_limb_traces(synth_mv), FS)
    assert c.score is not None and c.score > 0.9
    assert set(c.checks) >= {"einthoven", "avr", "avl", "avf", "goldberger_sum"}


def test_swapped_leads_break_the_identities(synth_mv):
    assert limb_consistency(_limb_traces(synth_mv, swap=True), FS).score < 0.7          # validate.limb_warn


def test_no_simultaneous_samples_no_score(synth_mv):
    tr = [("I", Trace.from_array(synth_mv["I"][:1250], start_s=0.0, fs=FS, quality=1.0)),
          ("II", Trace.from_array(synth_mv["II"][1250:2500], start_s=2.5, fs=FS, quality=1.0)),
          ("III", Trace.from_array(synth_mv["III"][:1250], start_s=5.0, fs=FS, quality=1.0))]
    assert limb_consistency(tr, FS).score is None


# --- end to end ---------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["3x4+rhythm", "6x2", "12x1"])
def test_layouts_are_read_and_match_the_truth(synth_mv, layout, tmp_path):
    spec = get_layout(layout)
    img = _page(synth_mv, layout=layout, header=False)
    dbg = ScanDebug()
    res = scan(_encode(img), hints=HINTS, debug=dbg, artifacts_dir=tmp_path)
    assert res.status == "ok", (res.warnings, res.reject_reasons)
    assert res.layout.name == layout
    assert set(res.leads) == {ld for row in spec.rows for ld in row}
    rs = []
    for ld, tr in res.leads.items():
        a = tr.array()
        i0 = int(round(tr.start_s * FS))
        ref = synth_mv[ld][i0:i0 + len(a)]
        ok = np.isfinite(a)
        rs.append(np.corrcoef(a[ok], ref[ok])[0, 1])
    assert np.median(rs) > 0.9 and min(rs) > 0.75          # bench clean_scan: r 0.956
    assert res.quality.limb_lead_consistency is None or res.quality.limb_lead_consistency > 0.7
    for name in ("rectified.png", "overlay.png", "mask.png", "overlay_preview.jpg"):
        assert (tmp_path / name).exists()


def test_steep_flanks_are_read_across_the_pen_not_down_the_column():
    """SD-058: a narrow R drawn with a 5-px pen merges into a blob near the tip; column-run
    middles sat ~0.2 mV low on the flanks (the tip itself comes from the spike rule)."""
    from ecgscan.vectorize.trace import column_runs, refine_steep

    xs = np.linspace(0, 120, 4801)
    curve = 150 - 110 * np.exp(-0.5 * ((xs - 60) / 3.5) ** 2)     # 110 px tall, flanks ~6 columns apart
    img = np.zeros((200, 121), np.uint8)
    pts = np.stack([xs * 16, curve * 16], 1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, 255, 5, lineType=cv2.LINE_AA, shift=4)
    ink = img > 127
    runs = [column_runs(ink[:, c]) for c in range(121)]
    mid = np.array([(r[0][0] + r[0][1] - 1) / 2 if r else np.nan for r in runs])
    truth = 150 - 110 * np.exp(-0.5 * ((np.arange(121) - 60) / 3.5) ** 2)
    mid[60] = truth[60]                                            # the tip, as _spike_columns gives it
    refined = refine_steep(mid, ink, 0, 5.0)
    flank = np.r_[54:60, 61:67]
    err_mid = np.sqrt(np.mean((mid[flank] - truth[flank]) ** 2))
    err_ref = np.sqrt(np.mean((refined[flank] - truth[flank]) ** 2))
    assert err_ref < 0.4 * err_mid, (err_mid, err_ref)            # 19 -> 5 px
    assert np.array_equal(refined[:40], mid[:40])                  # flat stretches untouched
