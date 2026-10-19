"""Photos of a device's own screen (D120): lit panel, grey grid, N leads as rows, ~2 s."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.screen import (
    CHEST_6,
    LIMB_6,
    digitize_screen,
    is_device_screen,
    render_device_screen,
)
from cardiolens.layer1_ingestion.strip import digitize_auto_to_signal
from cardiolens.pipeline import run_pipeline


@pytest.fixture(scope="module")
def sig():
    return generate_synthetic_ecg(heart_rate_bpm=95, duration_s=10.0, seed=3)


def _r(a, b):
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1])


def test_chest_screen_is_read_and_ordered(sig, tmp_path):
    path = render_device_screen(sig, tmp_path / "s.png", leads=CHEST_6)
    import cv2

    assert is_device_screen(cv2.imread(path))
    traces = digitize_screen(path)
    assert list(traces) == CHEST_6                     # not limb: rows 1-3 break Einthoven
    n = int(2.5 * sig.sampling_rate_hz)
    rs = [_r(sig.lead(ln)[:n], traces[ln]) for ln in CHEST_6]
    assert min(rs) > 0.85, rs


def test_limb_screen_is_recognised_by_einthoven(sig, tmp_path):
    assert list(digitize_screen(render_device_screen(sig, tmp_path / "l.png", leads=LIMB_6))) == LIMB_6


def test_screen_photo_through_the_pipeline(sig, tmp_path):
    s = digitize_auto_to_signal(render_device_screen(sig, tmp_path / "p.png", leads=CHEST_6))
    assert set(s.provenance.lead_windows) == set(CHEST_6)
    r = run_pipeline(s, mode="A")
    assert r.measurements.heart_rate_bpm == pytest.approx(95, abs=4)
    assert "screen_photo_limited" in r.findings.codes()


def test_paper_page_is_not_a_screen(sig, tmp_path):
    import cv2

    from cardiolens.layer1_ingestion.digitize import render_ecg_page

    path = str(tmp_path / "page.png")
    render_ecg_page(sig, path)
    assert not is_device_screen(cv2.imread(path))
