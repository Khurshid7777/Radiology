"""SD-059: 3-channel strip recorder (four groups of ~4 s with blank gaps, one start pulse)."""

import numpy as np
import pytest

from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.bench.sources import derive_limb_leads
from ecgscan.calibration.pulse import find_pulses, widen_strokes
from ecgscan.layout import get_layout
from ecgscan.layout.rows import long_stroke_mask
from ecgscan.pipeline import ScanDebug, ScanHints, scan

cv2 = pytest.importorskip("cv2")
STRIP = "3x4 strip"


@pytest.fixture(scope="module")
def strip():
    from cardiolens.layer1_ingestion import generate_synthetic_ecg

    s = generate_synthetic_ecg(heart_rate_bpm=69, seed=7, duration_s=17)
    mv = derive_limb_leads({n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names})
    img, gt = render_page(mv, 500.0, get_layout(STRIP), RenderSpec(px_per_mm=10.0, header=False))
    return img, gt


def _png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def test_registry_geometry():
    spec = get_layout(STRIP)
    assert spec.n_columns == 4 and spec.n_trace_rows == 3 and spec.start_pulse_required
    assert spec.column_offset_mm(1, 25) == pytest.approx(110.0)          # 100 mm + 10 mm gap
    assert spec.width_mm(25) == pytest.approx(430.0)
    assert get_layout("3x4+rhythm").column_offset_mm(1, 25) == pytest.approx(62.5)   # pages without gaps


def test_whole_strip_is_read_group_by_group(strip):
    img, gt = strip
    dbg = ScanDebug()
    res = scan(_png(img), debug=dbg, hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
    assert res.status != "rejected", res.reject_reasons
    assert res.layout.name == STRIP and dbg.frame.groups_found == 4
    assert dbg.frame.implied_speed_mm_s == pytest.approx(25, rel=0.03)
    rs = []
    for tr in gt.traces:
        a = res.leads[tr.lead].array()
        n = min(len(a), len(tr.samples_mv))
        ok = np.isfinite(a[:n])
        rs.append(np.corrcoef(a[:n][ok], tr.samples_mv[:n][ok])[0, 1])
    assert np.median(rs) > 0.95 and min(rs) > 0.85, rs                   # III is the low-amplitude one


@pytest.mark.parametrize("cut", ["start_group", "pulse_only"])
def test_a_strip_not_seen_whole_is_refused(strip, cut):
    """Naming groups by position needs the start: a photo that starts at V1-V3 would
    otherwise label them I-III."""
    img, _ = strip
    part = img[:, 1100:] if cut == "start_group" else img[:, 200:]       # group 1 gone / only the pulse gone
    res = scan(_png(part))
    assert res.status == "rejected" and res.reject_reasons == ["PAGE_INCOMPLETE"]


def test_broken_trace_is_one_row_and_its_label_does_not_move_the_start():
    ink = np.zeros((200, 900), bool)
    y = 100
    for a, b in ((100, 280), (285, 735), (740, 880)):                    # 18 + 45 + 14 mm with 0.5-mm breaks
        ink[y - 1:y + 2, a:b] = True
    ink[70:100, 300:302] = True                                          # a 3-mm complex
    ink[96:104, 82:86] = True                                            # a lead label's letter 3 mm before
    plain = long_stroke_mask(ink, 10.0, min_height_mm=2.0)
    bridged = long_stroke_mask(ink, 10.0, min_height_mm=2.0, bridge_mm=1.0)
    assert not plain[y, 100:280].any()                                   # the first 18 mm were dropped
    assert bridged[y, 100:280].all() and not bridged[96:104, 82:86].any()


def test_thin_slanted_broken_pulse_is_found_on_retry():
    m = np.zeros((300, 200), bool)
    for yy in range(100, 205):                                           # left edge drifts 3 px, with breaks
        if 140 <= yy < 146:
            continue
        m[yy, 50 + (yy - 100) // 20] = True
    for yy in range(100, 205):                                           # right edge drifts too
        if 150 <= yy < 155:
            continue
        m[yy, 115 + (yy - 100) // 25] = True
    m[100, 50:118] = True
    m[204, 30:52] = m[204, 116:140] = True
    assert find_pulses(m, 10.0) == []
    ps = find_pulses(widen_strokes(m, 5, 7), 10.0)                        # pipeline.yaml retry_widen_px / retry_close_px
    assert len(ps) == 1 and ps[0].height_mm == pytest.approx(10.4, abs=0.4)


def test_a_pulse_floating_below_its_row_is_not_another_row(strip):
    """The real strip's pulse sits 4.1 mm below row II; its bottom made a 4th row and a
    3x4+rhythm page with every lead mislabelled."""
    from ecgscan.layout.detect import detect_layout
    from ecgscan.segmentation import ClassicSegmenter

    img, gt = strip
    ink = ClassicSegmenter().segment(img, 10.0).ink
    row2 = sorted({t.baseline_px for t in gt.traces})[1]
    guess = detect_layout(ink, 10.0, pulse_bottoms=[int(row2 + 41)], stroke_min_height_mm=2.0)
    assert guess.name == STRIP and len(guess.baselines) == 3
