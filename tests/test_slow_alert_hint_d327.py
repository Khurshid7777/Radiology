"""D327: the severe-bradycardia and pause alerts name sinus node dysfunction or AV block; not under pacing."""
import pytest

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.safety.critical import SLOW_HINT, scan_critical


def _m(hr, max_rr):
    m = Measurements()
    m.heart_rate_bpm = hr
    m.max_rr_ms = max_rr
    return m


def test_slow_alerts_carry_the_hint():
    r = scan_critical(FindingSet(findings=[]), _m(30.0, 3500.0))
    by = {a.code: a for a in r.alerts}
    assert set(by) == {"severe_bradycardia", "ventricular_pause"}
    b, p = by["severe_bradycardia"], by["ventricular_pause"]
    assert b.label_ru.endswith(SLOW_HINT[0]) and b.label_uz.endswith(SLOW_HINT[1])
    assert p.label_ru == "Значимая пауза / асистолия"          # said once, on the bradycardia alert
    only_pause = {a.code: a for a in scan_critical(FindingSet(findings=[]), _m(55.0, 3500.0)).alerts}
    assert only_pause["ventricular_pause"].label_ru.endswith(SLOW_HINT[0])
    assert by["severe_bradycardia"].label_ru.startswith("Тяжёлая брадикардия — возможна дисфункция синусового узла")



@pytest.mark.parametrize("code", ["paced_rhythm", "complete_av_block", "probable_av_block_2to1",
                                  "slow_wide_escape_rhythm"])
def test_no_hint_under_pacing_or_a_named_av_block(code):
    fs = FindingSet(findings=[Finding(code=code, group="rhythm", label_ru="x", label_uz="x", evidence=[],
                                      confidence=0.9)])
    r = scan_critical(fs, _m(30.0, 3500.0))
    labels = {a.label_ru for a in r.alerts if a.code in ("severe_bradycardia", "ventricular_pause")}
    assert labels == {"Тяжёлая брадикардия", "Значимая пауза / асистолия"}


def test_other_alerts_untouched():
    assert scan_critical(FindingSet(findings=[]), _m(70.0, 900.0)).alerts == []


def test_quality_note_goes_before_the_hint():
    from cardiolens.pipeline import _before_hint
    note = " — сигнал низкого качества, проверить по записи"
    assert _before_hint("Тяжёлая брадикардия" + SLOW_HINT[0], note, SLOW_HINT[0]) == "Тяжёлая брадикардия" + note + SLOW_HINT[0]
    assert _before_hint("Полная АВ-блокада", note, SLOW_HINT[0]) == "Полная АВ-блокада" + note
