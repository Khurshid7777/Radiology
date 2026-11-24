"""SD-140: the pure parts of the period diagnostics (groups, autocorrelation features, quantiles)."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("period_diag", Path(__file__).parents[2] / "tools" / "scanner" / "period_diag.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


def test_groups_by_the_ratio_of_the_two_scales():
    assert pd.group_of(None) is None and pd.group_of(2.0) == "doubled" and pd.group_of(1.7) == "doubled"
    assert pd.group_of(1.0) == "normal" and pd.group_of(0.85) == "normal" and pd.group_of(1.2) == "normal"
    assert pd.group_of(0.5) == "half" and pd.group_of(1.4) is None and pd.group_of(0.7) is None


def test_features_of_a_clean_period_and_of_its_double():
    x = np.arange(2000)
    prof = np.cos(2 * np.pi * x / 20.0)[None, :]                     # one band, a period of 20 px
    a = pd.acorr(prof, 800)
    f = pd.features(a, 20.0)
    assert f["p_x2"] > 0.9 and f["p_x3"] > 0.9 and f["p_over_2"] < -0.9 and f["p_x1_5"] < -0.9
    g = pd.features(a, 40.0)                                         # the doubled reading: 1.5 P is a multiple of the true period
    assert g["p_x1_5"] > 0.9 and g["p_over_2"] > 0.9
    assert pd.features(np.zeros(10), 3.0) == {}


def test_report_keeps_only_group_quantiles():
    rows = [{"group": "doubled", "p_x1_5": 0.7, "check_fill": 1.8}, {"group": "normal", "p_x1_5": -0.4, "check_fill": 0.9},
            {"group": "normal", "p_x1_5": -0.3, "check_fill": 0.86}]
    r = pd.report(rows)
    assert r["doubled"]["n"] == 1 and r["normal"]["n"] == 2 and r["half"]["n"] == 0
    assert r["normal"]["p_x1_5"]["median"] == -0.3 and r["doubled"]["check_fill"]["median"] == 1.8
    assert "phone" not in json_dump(r)


def json_dump(r):
    import json

    return json.dumps(r)
