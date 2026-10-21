"""SD-060: the rhythm strip against the panels recorded at the same time."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from ecgscan.contract import Trace
from ecgscan.contract.cardiolens_adapter import to_signal_representation
from ecgscan.contract.models import RHYTHM_MISMATCH
from ecgscan.validate import rhythm_check

FS = 500.0


@pytest.fixture(scope="module")
def mv():
    from ecgscan.bench.sources import derive_limb_leads

    s = generate_synthetic_ecg(heart_rate_bpm=72, seed=9)
    return derive_limb_leads({n: s.lead(n).astype(np.float64) / 1000.0 for n in s.lead_names})


def _panels(mv):
    cols = (("I", "II", "III"), ("aVR", "aVL", "aVF"), ("V1", "V2", "V3"), ("V4", "V5", "V6"))
    out = {}
    for c, leads in enumerate(cols):
        a, b = int(c * 2.5 * FS), int((c + 1) * 2.5 * FS)
        for ld in leads:
            out[ld] = Trace.from_array(mv[ld][a:b], start_s=c * 2.5, fs=FS, lead=ld, quality=1.0)
    return out


def test_a_faithful_strip_agrees_with_panel_ii_and_goldberger(mv):
    checks = rhythm_check(_panels(mv), Trace.from_array(mv["II"], start_s=0, fs=FS, lead="II", quality=1.0), FS)
    assert set(checks) == {"panel_II", "aVF_aVR"} and max(checks.values()) < 0.1


def test_a_strip_out_of_step_with_the_panels_is_unrelated(mv):
    """Glare read a strip that has nothing to do with the beats the panels show."""
    strip = Trace.from_array(np.roll(mv["II"], 210), start_s=0, fs=FS, lead="II", quality=1.0)   # half an RR off
    checks = rhythm_check(_panels(mv), strip, FS)
    assert min(checks.values()) >= 0.9                               # pipeline.yaml rhythm_unrelated


def test_a_flagged_strip_is_not_measured(mv, result_factory):
    panels = _panels(mv)
    strip = Trace.from_array(mv["II"] * 0 + np.sin(np.arange(5000) / 30), start_s=0, fs=FS, lead="II", quality=0.0)
    strip.flags = [RHYTHM_MISMATCH]
    res = result_factory(status="needs_confirmation", leads=panels, rhythm=[strip])
    sig = to_signal_representation(res)
    ii = sig.lead_names.index("II")
    assert sig.provenance.lead_windows["II"] == (0.0, 2.5)          # the panel only
    assert np.allclose(sig.signal[ii, :1250], panels["II"].array() * 1000, atol=1)
