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
    """gain_5-004 as the scanner read it: synthetic RBBB at 5 mm/mV, traces r 0.98, but the
    QRS read 127 ms against 180 ms on the exact signal (the R' hump fell below threshold)."""
    return ScanResult.model_validate_json(gzip.decompress(FIX.read_bytes()))


def test_rbbb_width_hinges_on_the_hump_threshold(rbbb_scan):
    c = measurement_check(rbbb_scan, hump_scale=0.667)
    assert c.qrs_ms == pytest.approx(127.0, abs=6.0)
    assert c.qrs_ms_low_hump - c.qrs_ms > 20                 # 178: the R' counted


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
