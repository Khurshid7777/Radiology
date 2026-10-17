"""RU/UZ strings (SD-016): same keys, no empty text, every status/reject/warning code covered."""

import re

import pytest

from ecgscan.contract.models import REJECT_CODES, WARNING_CODES
from ecgscan.i18n import LANGS, strings, t


def test_languages_have_the_same_keys_and_placeholders():
    ru, uz = strings("ru"), strings("uz")
    assert set(ru) == set(uz)
    for k in ru:
        assert ru[k].strip() and uz[k].strip(), k
        assert set(re.findall(r"{(\w+)}", ru[k])) == set(re.findall(r"{(\w+)}", uz[k])), k


@pytest.mark.parametrize("lang", LANGS)
def test_every_code_and_status_has_a_text(lang):
    keys = set(strings(lang))
    assert {f"reject.{c}" for c in REJECT_CODES} <= keys
    assert {f"warning.{c}" for c in WARNING_CODES} <= keys
    assert {f"status.{s}" for s in ("ok", "needs_confirmation", "rejected", "queued", "processing", "failed")} <= keys


def test_formatting_and_unknown_language():
    assert "50" in t("result.calibration", "uz", speed=50, gain=10)
    with pytest.raises(ValueError):
        strings("en")
