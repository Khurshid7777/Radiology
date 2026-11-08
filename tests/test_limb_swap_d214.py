"""D214: limb-electrode swaps found by predicting I and II from the chest leads."""

import warnings

import numpy as np
import pytest

from cardiolens.api.service import signal_from_case
from cardiolens.layer2_preprocess.quality import detect_limb_swap, swap_limb_leads

warnings.filterwarnings("ignore")


def _swapped(sig, kind):
    s = sig.model_copy(deep=True)
    i, ii = swap_limb_leads(sig.lead("I").astype(float), sig.lead("II").astype(float), kind)
    new = {"I": i, "II": ii, "III": ii - i, "aVR": -(i + ii) / 2, "aVL": i - ii / 2, "aVF": ii - i / 2}
    arr = np.array(s.signal, dtype=float)
    for ld, v in new.items():
        arr[s.lead_names.index(ld)] = v
    s.signal = arr
    return s


def test_recorded_normal_is_not_flagged():
    assert detect_limb_swap(signal_from_case("normal")) is None


@pytest.mark.parametrize("kind", ["la_ra", "ra_ll"])
def test_swaps_are_named(kind):
    assert detect_limb_swap(_swapped(signal_from_case("normal"), kind)) == kind


def test_swap_helper_is_an_involution():
    sig = signal_from_case("normal")
    i, ii = sig.lead("I").astype(float), sig.lead("II").astype(float)
    for kind in ("la_ra", "ra_ll", "la_ll"):
        a, b = swap_limb_leads(*swap_limb_leads(i, ii, kind), kind)
        assert np.allclose(a, i) and np.allclose(b, ii)


def test_pipeline_reports_ra_ll():
    from cardiolens.pipeline import run_pipeline

    res = run_pipeline(_swapped(signal_from_case("normal"), "ra_ll"), mode="A")
    assert "lead_reversal_ra_ll" in res.findings.codes()
    assert "lead_reversal_la_ra" not in res.findings.codes()
