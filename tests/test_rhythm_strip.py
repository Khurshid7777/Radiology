"""3x4 + rhythm strips (D141): three rows of lead groups back to back, II below for the
whole record; photographed whole or in pieces."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.sequential import (
    RHYTHM_LOW_RES,
    digitize_rhythm_strip,
    render_rhythm_strip,
)
from cardiolens.layer1_ingestion.strip import digitize_auto_to_signal
from cardiolens.pipeline import run_pipeline


@pytest.fixture(scope="module")
def sig():
    s = generate_synthetic_ecg(heart_rate_bpm=80, duration_s=12.0, seed=3)
    x = np.array(s.signal, dtype=np.float32)
    lead = s.lead_names.index
    i, ii = x[lead("I")], x[lead("II")]
    x[lead("III")] = ii - i
    x[lead("aVR")] = -(i + ii) / 2
    x[lead("aVL")] = i - ii / 2
    x[lead("aVF")] = ii - i / 2
    return s.model_copy(update={"signal": x})


@pytest.mark.parametrize("window,leads", [((0, 10), {"I", "aVR", "V1", "V4"}), ((0, 4), {"I"}),
                                          ((3, 7.5), {"V1"})])
def test_groups_are_named(sig, tmp_path, window, leads):
    _, win = digitize_rhythm_strip(render_rhythm_strip(sig, tmp_path / "s.png", window_s=window))
    assert leads <= set(win) and "II" in win
    assert not ({"V1", "V4"} - leads) & set(win)


def test_whole_strip_through_the_pipeline(sig, tmp_path):
    s = digitize_auto_to_signal(render_rhythm_strip(sig, tmp_path / "s.png", px_per_mm=10))
    assert any(n.startswith("3x4+rhythm strip photo") for n in s.provenance.notes)
    assert len(s.provenance.lead_windows) == 12
    r = run_pipeline(s, mode="A")
    assert r.measurements.heart_rate_bpm == pytest.approx(80, rel=0.05)


def test_chest_groups_alone_are_not_named(sig, tmp_path):
    with pytest.raises(ValueError, match="cannot be named"):
        digitize_rhythm_strip(render_rhythm_strip(sig, tmp_path / "c.png", window_s=(6, 10)))


def test_coarse_strip_is_refused_not_read_as_a_page(sig, tmp_path):
    """D141: the FK strip at ~5 px/mm read as a 3x4 page gave HR 42 / 55 / 89 per piece."""
    with pytest.raises(ValueError, match=RHYTHM_LOW_RES):
        digitize_auto_to_signal(render_rhythm_strip(sig, tmp_path / "k.png", px_per_mm=5))


@pytest.mark.parametrize("degrade", [dict(rotation_deg=2), dict(perspective=0.02, jpeg_quality=70)])
def test_photo_like_piece_is_named_or_refused_never_misread(sig, tmp_path, degrade):
    """D143: a 4-s piece the reader could not name went on to the screen and page readers,
    which read a rate of 55 as 35-100."""
    from cardiolens.layer1_ingestion.digitize import degrade_image

    src = render_rhythm_strip(sig, tmp_path / "p.png", window_s=(0, 4))
    path = degrade_image(src, tmp_path / "d.png", seed=3, **degrade)
    try:
        s = digitize_auto_to_signal(path)
    except ValueError:
        return
    assert any(n.startswith("3x4+rhythm strip photo") for n in s.provenance.notes)
    assert run_pipeline(s, mode="A").measurements.heart_rate_bpm == pytest.approx(80, rel=0.05)
