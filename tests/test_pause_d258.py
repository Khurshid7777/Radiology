"""D258: a long RR in AF is not a sinus pause; a compensatory pause after a premature beat is not named; a true
sinus pause (>= 2 s) still is."""
from cardiolens.config import load_app_config
from cardiolens.contracts.features import Measurements
from cardiolens.pipeline import _pause_finding

PC = load_app_config().pause


def _m(rr_ms, **kw):
    rp, t = [0], 0
    for x in rr_ms:
        t += int(x / 2)          # samples at 500 Hz
        rp.append(t)
    mx = max(rr_ms)
    med = sorted(rr_ms)[len(rr_ms) // 2]
    return Measurements(r_peaks_samples=rp, max_rr_ms=mx, pause_ratio=round(mx / med, 2), **kw)


def test_true_sinus_pause_is_named():
    f = _pause_finding(_m([800, 800, 800, 2400, 800, 800]), PC, {"sinus_rhythm"}, set())
    assert f is not None and f.code == "sinus_pause" and not f.critical


def test_long_rr_in_af_is_a_ventricular_pause():
    f = _pause_finding(_m([600, 900, 700, 2200, 650, 800]), PC, {"atrial_fibrillation"}, set())
    assert f is not None and f.code == "rr_pause_af"
    assert _pause_finding(_m([600, 900, 700, 1500, 650, 800]), PC, {"atrial_fibrillation"}, set()) is None


def test_compensatory_pause_after_a_pvc_is_not_named():
    m = _m([800, 800, 480, 1600, 800, 800])          # beat 3 is the PVC, the long RR follows it
    assert _pause_finding(m, PC, {"ventricular_ectopy"}, {3}) is None
