"""D324: VF-like waves at a harmonic of a regular R train — same alarm, wording 'regular tachycardia, exclude VT / VF'."""
import numpy as np

from cardiolens.arrhythmia.vf import vf_is_harmonic
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.safety.critical import VF_REGULAR_ALERT, scan_critical


def _mts(f, n=6):
    return [{"f_dom": f, "conc": 0.8, "leak": 0.4, "rms": 300.0, "crest": 2.2}] * n


def test_second_harmonic_of_a_regular_train():
    fs = 500.0
    r = np.arange(17) * 0.556 * fs                      # 108 /min, regular
    assert vf_is_harmonic(_mts(3.5), r, fs)             # 3.5 Hz ~ 2 x 1.8 Hz
    assert not vf_is_harmonic(_mts(1.8), r, fs)         # 1st harmonic: every wave a beat (VFL) - untouched
    assert not vf_is_harmonic(_mts(5.0), r, fs)


def test_irregular_or_short_train_is_not_harmonic():
    fs = 500.0
    rng = np.random.default_rng(0)
    r = np.cumsum(rng.uniform(0.3, 0.8, 17)) * fs        # irregular, as on VF
    assert not vf_is_harmonic(_mts(3.5), r, fs)
    assert not vf_is_harmonic(_mts(3.5), np.arange(4) * 0.556 * fs, fs)
    assert not vf_is_harmonic([None, None], np.arange(17) * 0.556 * fs, fs)


def _vf_finding(ev):
    return Finding(code="ventricular_fibrillation", group="rhythm", label_ru="x", label_uz="x",
                   evidence=[ev], confidence=0.5, critical=True)


def test_alert_wording_follows_the_pattern_and_the_alarm_stays():
    from cardiolens.contracts.features import Measurements
    m = Measurements()
    reg = scan_critical(FindingSet(findings=[_vf_finding(Evidence(feature="vf_pattern", value="regular_harmonic"))]), m)
    chaos = scan_critical(FindingSet(findings=[_vf_finding(Evidence(feature="vf_detector", value="soft"))]), m)
    assert [a.code for a in reg.alerts] == [a.code for a in chaos.alerts] == ["ventricular_fibrillation"]
    assert (reg.alerts[0].label_ru, reg.alerts[0].label_uz) == VF_REGULAR_ALERT
    assert chaos.alerts[0].label_ru.startswith("Фибрилляция")
