"""SD-150: lead labels on a strip photo: only the twelve names, never a guess; groups named by two labels in a column."""

import numpy as np
import pytest

from ecgscan.layout.labels import GROUPS, Label, groups_seen, leads_in, normalise, read_lead_labels


def test_only_lead_names_are_kept_and_nothing_is_guessed():
    assert leads_in("aVR") == ["aVR"] and leads_in("AVL") == ["aVL"] and leads_in("avf") == ["aVF"]
    assert leads_in("V 4") == ["V4"] and leads_in("v6") == ["V6"] and leads_in("III") == ["III"]
    assert leads_in("aVR aVL") == ["aVR", "aVL"]
    for junk in ("14", "V7", "Name:TEST PATIENT", "25mm/s", "X:M", "IV", "Vi", "aVX", "II5"):
        assert leads_in(junk) == [], junk
    assert normalise("v 2") == "V2" and normalise("Avr") == "aVR" and normalise("X") is None


def fake_engine(boxes):
    def run(img):
        return [[[[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]], text, score]
                for text, x, y, score in boxes], None
    return run


def test_labels_are_placed_on_the_photo_and_weak_or_mixed_boxes_dropped():
    img = np.zeros((100, 200, 3), np.uint8)
    eng = fake_engine([("aVR", 100, 40, 0.9), ("aVL", 100, 80, 0.4), ("aVR aVL", 300, 40, 0.9), ("14", 20, 20, 0.9)])
    got = read_lead_labels(img, engine=eng, upscale=2.0)
    assert got == [Label("aVR", 50.0, 20.0, 0.9)]                       # weak, two-in-one and non-lead boxes dropped


def test_a_group_needs_two_labels_in_one_column_in_row_order():
    lb = [Label("V4", 500, 100, 0.9), Label("V5", 503, 300, 0.9), Label("aVR", 100, 100, 0.9)]
    assert groups_seen(lb, column_tol_px=20) == [(GROUPS[3], 501.5)]       # aVR alone names nothing
    assert groups_seen([Label("V4", 500, 100, 0.9), Label("V5", 560, 300, 0.9)], column_tol_px=20) == []
    assert groups_seen([Label("V4", 500, 300, 0.9), Label("V5", 500, 100, 0.9)], column_tol_px=20) == []   # upside down


@pytest.mark.skipif(not __import__("importlib").util.find_spec("rapidocr_onnxruntime"), reason="no OCR engine")
def test_a_rendered_strip_part_names_its_group():
    pytest.importorskip("cv2")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("sp", Path(__file__).parents[2] / "tools" / "scanner" / "strip_parts.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    page, gt = sp.strip_page(701)
    parts = sp.cut(page, gt, 4)
    named = set()
    for part, _ in parts:
        img = sp.photo(part, 5)
        named |= {g for g, _ in groups_seen(read_lead_labels(img), column_tol_px=0.05 * img.shape[1])}
    assert named and named <= set(GROUPS) and named & {GROUPS[1], GROUPS[2], GROUPS[3]}
