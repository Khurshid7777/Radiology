"""D241: the long-QT finding stays from QTc > 450 / 460; the critical alert only from QTc >= 500 ms."""
from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.safety.critical import LONG_QT_CRITICAL_MS, scan_critical


def _fs(qtc: float) -> FindingSet:
    return FindingSet(findings=[Finding(code="long_qt", group="interval", label_ru="QT", label_uz="QT",
                                        evidence=[Evidence(feature="qtc_ms", value=str(qtc))], confidence=0.85)])


def test_moderate_prolongation_is_a_finding_without_alert():
    fs = _fs(472.0)
    res = scan_critical(fs, Measurements(heart_rate_bpm=70))
    assert "long_qt" not in [a.code for a in res.alerts]
    assert "long_qt" in fs.codes() and not fs.by_code("long_qt").critical


def test_qtc_from_500_raises_the_alert():
    fs = _fs(LONG_QT_CRITICAL_MS + 3)
    res = scan_critical(fs, Measurements(heart_rate_bpm=70))
    assert "long_qt" in [a.code for a in res.alerts] and fs.by_code("long_qt").critical
