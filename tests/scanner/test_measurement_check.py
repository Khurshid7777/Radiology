"""SD-057: a QRS width that hinges on a small terminal R' is not returned as ``ok``."""

import gzip
from pathlib import Path

import numpy as np
import pytest

from ecgscan.contract import ScanResult
from ecgscan.validate import measurement_check

FIX = Path(__file__).parent / "fixtures" / "rbbb_gain5_scan.json.gz"


@pytest.fixture(scope="module")
def rbbb_scan() -> ScanResult:
    """gain_5-004 as the scanner read it: synthetic RBBB at 5 mm/mV, traces r 0.98, 180 ms on the
    exact signal (before D236 the R' hump fell below threshold and the QRS read 127 ms)."""
    return ScanResult.model_validate_json(gzip.decompress(FIX.read_bytes()))


def test_rbbb_width_no_longer_hinges_on_the_hump_threshold(rbbb_scan):
    """Before CardioLens D236 this scan read 127 ms at the default hump threshold and 178 with
    the R' counted. D236 retries a wide beat with a longer search and keeps the true 180 ms at
    every threshold: the width is stable, so QRS_UNSTABLE does not fire on it (the flag itself
    is covered by ``test_scan_flags_an_unstable_qrs``)."""
    for scale in (None, 0.667, 1.0):
        c = measurement_check(rbbb_scan, hump_scale=scale)
        assert c.qrs_ms == pytest.approx(180.0, abs=6.0)
        assert c.qrs_ms_low_hump is None or abs(c.qrs_ms_low_hump - c.qrs_ms) <= 20


def test_cardiolens_default_is_unchanged(rbbb_scan):
    """The override is per call; CardioLens itself measures as before."""
    from cardiolens.layer3_measure.delineation import QRS_HUMP_FRACTION

    a = measurement_check(rbbb_scan)
    b = measurement_check(rbbb_scan, hump_scale=1.0)
    assert QRS_HUMP_FRACTION == 0.15 and a.qrs_ms == b.qrs_ms == b.qrs_ms_low_hump


def test_scan_flags_an_unstable_qrs(monkeypatch, synth_mv):
    import cv2

    import ecgscan.pipeline as P
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout import get_layout
    from ecgscan.validate import MeasurementCheck

    monkeypatch.setattr(P, "measurement_check",
                        lambda res, hump_scale=None: MeasurementCheck(True, 127.0, 178.0))
    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    res = P.scan(cv2.imencode(".png", img)[1].tobytes(), hints=P.ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
    assert res.status == "needs_confirmation" and "QRS_UNSTABLE" in res.warnings
    assert np.isfinite(res.leads["II"].array()).mean() > 0.9   # the traces are still returned


def test_pr_on_fallback_onsets_is_unstable():
    """SD-110: a PR with the P onset refined on under half the beats rests on the fallback onsets."""
    from ecgscan.validate import MeasurementCheck, pr_unstable

    assert pr_unstable(MeasurementCheck(pr_ms=160.0, p_onset_ratio=0.0), 0.5)
    assert pr_unstable(MeasurementCheck(pr_ms=160.0, p_onset_ratio=None), 0.5)
    assert not pr_unstable(MeasurementCheck(pr_ms=160.0, p_onset_ratio=0.8), 0.5)
    assert not pr_unstable(MeasurementCheck(pr_ms=None, p_onset_ratio=0.0), 0.5)    # no PR given: nothing to flag
    assert not pr_unstable(MeasurementCheck(pr_ms=160.0, p_onset_ratio=0.0), None)  # switched off


def test_scan_flags_an_unstable_pr(monkeypatch, synth_mv):
    import cv2

    import ecgscan.pipeline as P
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.contract.models import ScanResult
    from ecgscan.layout import get_layout
    from ecgscan.validate import MeasurementCheck

    img, _ = render_page(synth_mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=10.0, header=False))
    data, hints = cv2.imencode(".png", img)[1].tobytes(), P.ScanHints(paper_speed_mm_s=25, gain_mm_mv=10)
    monkeypatch.setattr(P, "measurement_check",
                        lambda res, hump_scale=None: MeasurementCheck(True, 90.0, 90.0, pr_ms=150.0, p_onset_ratio=0.9))
    assert P.scan(data, hints=hints).status == "ok"
    monkeypatch.setattr(P, "measurement_check",
                        lambda res, hump_scale=None: MeasurementCheck(True, 90.0, 90.0, pr_ms=150.0, p_onset_ratio=0.2))
    res = P.scan(data, hints=hints)
    assert res.status == "needs_confirmation" and "PR_UNSTABLE" in res.warnings
    ScanResult.model_validate_json(res.model_dump_json())                           # the code is in the contract


def test_pr_counts_in_the_bench_bad_definition():
    import json
    from pathlib import Path

    import yaml

    from ecgscan.bench.metrics import INTERVALS

    acc = yaml.safe_load(Path("configs/scanner/bench.yaml").read_text())["acceptance"]
    assert "pr_ms" in acc["bad_intervals"] and set(acc["bad_intervals"]) <= set(INTERVALS)
    for d in ("configs/scanner/i18n", "mobile/ecg_scanner/assets/i18n"):
        for lang in ("ru", "uz"):
            assert json.loads(Path(f"{d}/{lang}.json").read_text())["warning.PR_UNSTABLE"]
