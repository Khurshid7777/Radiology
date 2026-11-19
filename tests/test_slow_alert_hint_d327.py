"""D327: the severe-bradycardia and pause alerts name sinus node dysfunction or AV block; not under pacing."""
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
    for a in by.values():
        assert a.label_ru.endswith(SLOW_HINT[0]) and a.label_uz.endswith(SLOW_HINT[1])
    assert by["severe_bradycardia"].label_ru.startswith("Тяжёлая брадикардия — возможна дисфункция синусового узла")


def test_no_hint_under_pacing_and_other_alerts_untouched():
    paced = FindingSet(findings=[Finding(code="paced_rhythm", group="rhythm", label_ru="x", label_uz="x",
                                         evidence=[], confidence=0.9)])
    r = scan_critical(paced, _m(30.0, 3500.0))
    assert {a.label_ru for a in r.alerts} == {"Тяжёлая брадикардия", "Значимая пауза / асистолия"}
    assert scan_critical(FindingSet(findings=[]), _m(70.0, 900.0)).alerts == []
