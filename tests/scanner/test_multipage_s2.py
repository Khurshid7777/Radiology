"""S2 (TZ 18.4): a study on several sheets — a rhythm strip cut in two, stitched by the overlap."""

import dataclasses

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.bench.sources import derive_limb_leads
from ecgscan.layout import get_layout
from ecgscan.multipage import find_overlap, r_peaks
from ecgscan.pipeline import ScanHints, scan

cv2 = pytest.importorskip("cv2")
FS = 500.0


def _sheets(cuts, *, total_s=16.0, hr=72, jitter=30.0, seed=5, ppm=10.0):
    s = generate_synthetic_ecg(heart_rate_bpm=hr, seed=seed, duration_s=total_s, rr_jitter_ms=jitter)
    mv = derive_limb_leads({k: s.lead(k).astype(np.float64) / 1000.0 for k in s.lead_names})
    page, gt = render_page(mv, FS, dataclasses.replace(get_layout("single"), panel_s=total_s), RenderSpec(px_per_mm=ppm))
    x0 = gt.traces[0].x0_px
    out = []
    for k, (a, b) in enumerate(cuts):
        left = 0 if k == 0 else int(x0 + a * 25 * ppm)
        out.append(cv2.imencode(".png", np.ascontiguousarray(page[:, left:int(x0 + b * 25 * ppm)]))[1].tobytes())
    return mv, out


def test_overlap_is_found_by_the_beats_and_a_regular_rhythm_is_not_guessed():
    s = generate_synthetic_ecg(heart_rate_bpm=70, seed=3, duration_s=12.0, rr_jitter_ms=30.0)
    x = s.lead("II").astype(np.float64) / 1000.0
    a, b = x[:int(7.5 * FS)], x[int(4 * FS):]               # b repeats a's last 3.5 s (4 beats)
    lap, ambiguous = find_overlap(a, b, FS)
    assert lap == pytest.approx(3.5 * FS, abs=3) and not ambiguous
    assert find_overlap(x[:int(7 * FS)], x[int(5 * FS):], FS) == (None, False)   # 2 s: < 3 beats, not taken
    assert find_overlap(x[:int(6 * FS)], x[int(6 * FS):], FS)[0] is None      # cut end to end: no overlap
    flat = generate_synthetic_ecg(heart_rate_bpm=70, seed=3, duration_s=12.0).lead("II").astype(np.float64) / 1000.0
    lap, ambiguous = find_overlap(flat[:int(7.5 * FS)], flat[int(4 * FS):], FS)
    assert lap is None and ambiguous                          # every beat looks alike: not guessed


def test_a_strip_on_two_sheets_is_one_record_with_the_seam_checked(synth_mv):
    mv, sheets = _sheets(((0.0, 9.5), (6.0, 16.0)))          # 3.5 s overlap, as the app asks
    res = scan(sheets, hints=ScanHints(pages=True))
    assert "PAGES_STITCHED" in res.warnings and "SEAM_UNCERTAIN" not in res.warnings
    assert [p.index for p in res.pages] == [0, 1] and res.pages[0].overlap_with_next
    assert 3.0 <= res.pages[0].overlap_s <= 3.5 and res.pages[1].overlap_with_next is None   # less the cut edges
    assert len(res.source.parts_sha256) == 2
    tp, rp = r_peaks(mv["II"], FS), r_peaks(res.leads["II"].array(), FS)
    assert len(tp) == len(rp)                                  # no beat lost or doubled at the seam
    assert np.max(np.abs(np.diff(tp) - np.diff(rp))) / FS <= 0.020
    assert len(res.leads["II"].samples_mv) / FS == pytest.approx(16.0, abs=0.3)


def test_sheets_without_an_overlap_are_flagged_not_trusted():
    _, sheets = _sheets(((0.0, 8.5), (7.5, 16.0)), jitter=0.0)     # 1 s, regular: not verifiable
    res = scan(sheets, hints=ScanHints(pages=True))
    assert "SEAM_UNCERTAIN" in res.warnings and res.status == "needs_confirmation"
    assert res.pages[0].overlap_with_next is False


def test_a_list_without_pages_is_still_a_board_set(synth_mv):
    _, sheets = _sheets(((0.0, 9.0), (7.0, 16.0)))
    res = scan(sheets)                                         # SD-066: parts of one page on a board
    assert res.reject_reasons == ["BOARD_NOT_FOUND"] and res.pages is None
