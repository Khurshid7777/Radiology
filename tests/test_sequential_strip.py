"""Three-channel strips (D125): three rows, lead groups I-III, aVR-aVF, V1-V3, V4-V6 in turn,
photographed on a patterned desk, possibly turned."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.screen import combine_screen_signals
from cardiolens.layer1_ingestion.sequential import (
    SEQ_GROUPS,
    digitize_sequential,
    digitize_sequential_to_signal,
    is_sequential_strip,
    render_sequential_strip,
)
from cardiolens.layer1_ingestion.strip import digitize_auto_to_signal
from cardiolens.pipeline import run_pipeline


def _einthoven(sig):
    """The synthetic generator's limb leads are only roughly consistent; a strip's are exactly."""
    x = np.array(sig.signal, dtype=np.float32)
    lead = sig.lead_names.index
    i, ii = x[lead("I")], x[lead("II")]
    x[lead("III")] = ii - i
    x[lead("aVR")] = -(i + ii) / 2
    x[lead("aVL")] = i - ii / 2
    x[lead("aVF")] = ii - i / 2
    return sig.model_copy(update={"signal": x})


@pytest.fixture(scope="module")
def sig():
    return _einthoven(generate_synthetic_ecg(heart_rate_bpm=80, duration_s=16.0, seed=3))


@pytest.mark.parametrize("groups,first,turn", [(2, 0, 0), (2, 0, 1), (2, 0, 2), (3, 1, 0)])
def test_groups_are_found_and_named(sig, tmp_path, groups, first, turn):
    import cv2

    path = render_sequential_strip(sig, tmp_path / "s.png", groups=groups, first=first, turn=turn)
    assert is_sequential_strip(cv2.imread(path))
    out = digitize_sequential(path)
    assert [list(g) for g in out] == SEQ_GROUPS[first:first + groups]
    fs, fs_img = sig.sampling_rate_hz, digitize_sequential.last_calibration["fs"]
    n = int(2.0 * fs)
    for j, g in enumerate(out):
        for ln, tr in g.items():
            src = sig.lead(ln)[int((j * 4.0 + 0.1) * fs):][:n]
            got = np.interp(np.arange(n) / fs, np.arange(len(tr)) / fs_img, tr)
            assert np.corrcoef(src, got)[0, 1] > 0.6, (ln, turn)


def test_strip_runs_through_the_pipeline(sig, tmp_path):
    path = render_sequential_strip(sig, tmp_path / "s.png", groups=2)
    s = digitize_auto_to_signal(path)
    assert any(n.startswith("three-channel strip photo") for n in s.provenance.notes)
    assert set(s.provenance.lead_windows) == {"I", "II", "III", "aVR", "aVL", "aVF"}
    assert len(s.provenance.discontinuities_s) == 1
    r = run_pipeline(s, mode="A")
    assert abs(r.measurements.heart_rate_bpm - 80) <= 5
    assert not [a.code for a in r.safety.alerts]


def test_two_photos_of_one_strip_combine(sig, tmp_path):
    a = digitize_sequential_to_signal(render_sequential_strip(sig, tmp_path / "a.png", groups=2, first=0))
    b = digitize_sequential_to_signal(render_sequential_strip(sig, tmp_path / "b.png", groups=3, first=1))
    s = combine_screen_signals([a, b], drop_repeats=True)
    assert set(s.provenance.lead_windows) == {ln for g in SEQ_GROUPS for ln in g}
    # every group stays its own window: 1 cut inside a, 1 between the photos, 2 inside b
    assert len(s.provenance.discontinuities_s) == 4
