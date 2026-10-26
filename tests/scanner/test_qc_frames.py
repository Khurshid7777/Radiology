"""S1 (TZ addendum §18.14): the known-good / known-bad frame set and its acceptance count."""

from ecgscan.bench.qc_frames import BAD, CLASSES, evaluate, make_frame


def test_one_frame_per_class_is_refused_or_kept_for_its_defect():
    frames = [make_frame(cls, 500 + i) for i, cls in enumerate(CLASSES)]
    res = evaluate(frames)
    by = {f["cls"]: f["hints"] for f in res["frames"]}
    assert by["good"] == []
    assert "closer" in by["far"] and "frame" in by["cut"] and "steady" in by["blur"]
    assert "light" in by["dark"] and "level" in by["rotated"]
    assert all(by[c] for c in BAD)
    acc = res["summary"]["_acceptance"]
    assert acc["bad_refused"] == 1.0 and acc["good_refused"] == 0.0


def test_frames_are_reproducible():
    a, b = make_frame("shadow", 7), make_frame("shadow", 7)
    assert (a.rgb == b.rgb).all() and a.scale_to_capture == b.scale_to_capture
