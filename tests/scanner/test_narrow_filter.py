"""SD-151: a filter printed on the header that narrows the record (35/40 Hz high cut, 0.5-1 Hz low cut) is warned about."""

from ecgscan.config import load_yaml
from ecgscan.contract.models import WARNING_CODES, HeaderOcr
from ecgscan.i18n import strings
from ecgscan.pipeline import narrow_filter

Q = load_yaml("pipeline")["quality"]


def header(hp=None, lp=None):
    return HeaderOcr(engine="t", confidence=0.9, filter_hp_hz=hp, filter_lp_hz=lp)


def test_a_narrow_band_is_warned_and_a_full_one_is_not():
    assert narrow_filter(header(0.05, 35), Q) and narrow_filter(header(0.05, 40), Q)
    assert narrow_filter(header(1.0, 150), Q) and narrow_filter(header(1.0, None), Q)
    assert not narrow_filter(header(0.5, 150), Q)                      # 0.5 Hz is within the AHA limit for ST
    assert not narrow_filter(header(0.05, 150), Q) and not narrow_filter(header(0.05, 250), Q)
    assert not narrow_filter(header(0.67, 150), Q)                     # AHA's limit itself is allowed
    assert not narrow_filter(header(None, None), Q) and not narrow_filter(None, Q)


def test_the_code_and_its_words():
    assert "NARROW_FILTER" in WARNING_CODES
    assert Q["narrow_filter_lp_hz"] == 150.0 and Q["wide_filter_hp_hz"] == 0.67
    for lang in ("ru", "uz"):
        assert "ST" in strings(lang)["warning.NARROW_FILTER"]
