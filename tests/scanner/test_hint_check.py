"""SD-139: the pure parts of the frame-hint measurement (classification of outcomes, shares)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("hint_check", Path(__file__).parents[2] / "tools" / "scanner" / "hint_check.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


def row(hints, status, phone="photos_iphone", **kw):
    return {"phone": phone, "hints": hints, "status": status, "reasons": [], "ppm_check": None, "ppm_scan": None, **kw}


def test_outcome_classes():
    assert hc.outcome("ok") == "reads" and hc.outcome("needs_confirmation") == "confirm"
    assert hc.outcome("rejected") == "refused" and hc.outcome("anything else") == "refused"


def test_false_hints_and_misses_are_counted_per_hint():
    rows = [row(["closer"], "ok"), row(["closer", "frame"], "rejected"), row(["frame"], "ok"),
            row([], "ok"), row([], "rejected"), row([], "needs_confirmation")]
    r = hc.rates(rows)
    assert r["n"] == 6 and r["reads"] == 3
    assert r["closer"] == {"named": 2, "named_share": 0.3333, "false": 1, "false_share": 0.5}
    assert r["frame"]["named"] == 2 and r["frame"]["false"] == 1
    assert r["glare"] == {"named": 0, "named_share": 0.0, "false": 0, "false_share": None}
    assert r["any"]["named"] == 3 and r["any"]["false"] == 2
    assert r["no_hint"] == {"n": 3, "refused": 1, "refused_share": 0.3333}
    assert hc.rates([])["any"]["named_share"] is None


def test_report_groups_by_phone_and_reads_scale_ratios(tmp_path):
    import json

    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        row(["closer"], "ok", ppm_check=8.0, ppm_scan=10.0), row([], "rejected", "photos_samsung", reasons=["LAYOUT_UNKNOWN"]),
    ]))
    rep = hc.report([p])
    assert rep["all"]["n"] == 2 and rep["photos_iphone"]["n"] == 1 and rep["photos_samsung"]["n"] == 1
    assert rep["ppm_check_over_scan"]["median"] == 0.8 or rep["ppm_check_over_scan"]["n"] == 1
    assert rep["refusal_reasons"] == {"LAYOUT_UNKNOWN": 1}


def test_sd157_doubled_without_hints_and_the_weak_grid_rule():
    import numpy as np

    rows = [{"hints": [], "ppm_check": 20.0, "ppm_scan": 10.0}, {"hints": ["frame"], "ppm_check": 20.0, "ppm_scan": 10.0},
            {"hints": [], "ppm_check": 10.5, "ppm_scan": 10.0}, {"hints": [], "ppm_check": None, "ppm_scan": 10.0}]
    assert hc.doubled_silent(rows) == 1
    assert "grid" in hc.HINTS
    blank = np.full((600, 900, 3), 255, np.uint8)                  # no grid: no px/mm, so no 'grid' either
    hints, ppm, _, sec = hc.check_hints(blank, 1600, 0.15)
    assert ppm is None and "grid" not in hints and sec >= 0
