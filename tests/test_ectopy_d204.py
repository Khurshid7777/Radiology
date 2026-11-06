"""D204: supraventricular premature beats, atrial and ventricular bi-/trigeminy, ventricular
bigeminy found against a non-blended template, sinus arrhythmia, the "trifascicular" name."""

import numpy as np

from cardiolens.arrhythmia.beats import (
    detect_pac_beats,
    detect_pvc_beats,
    pac_findings,
    pvc_findings,
    sinus_arrhythmia_finding,
)
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import SignalRepresentation


def _train(rr_ms, odd=None, fs=500.0):
    t_r = np.cumsum([400.0] + list(rr_ms)) / 1000.0
    n = int((t_r[-1] + 0.5) * fs)
    x = np.zeros((3, n))
    tt = np.arange(n) / fs
    for k, t in enumerate(t_r):
        if odd and k in odd:
            x += 900 * (np.exp(-((tt - t + 0.03) / 0.03) ** 2) - np.exp(-((tt - t - 0.04) / 0.03) ** 2))
        else:
            x += 1000 * np.exp(-((tt - t) / 0.012) ** 2)
    sig = SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=fs,
                               lead_names=["II", "V1", "V5"], duration_s=n / fs, source="synthetic")
    return sig, (t_r * fs).astype(int)


def test_an_early_narrow_beat_is_a_pac_and_not_a_pvc():
    sig, rp = _train([800] * 5 + [560, 900] + [800] * 5)
    assert detect_pac_beats(sig, rp, 500.0) == [6]
    assert detect_pvc_beats(sig, rp, 500.0) == []


def test_a_wide_early_beat_is_left_to_the_pvc_rule():
    sig, rp = _train([800] * 5 + [560, 1040] + [800] * 5, odd={6})
    pvc = detect_pvc_beats(sig, rp, 500.0)
    assert pvc == [6] and detect_pac_beats(sig, rp, 500.0, skip=pvc) == []


def test_a_slow_drift_is_not_premature():
    rr = [800, 780, 750, 720, 690, 660, 690, 720, 750, 780, 800, 780]   # breathing, no jump
    sig, rp = _train(rr)
    assert detect_pac_beats(sig, rp, 500.0) == []


def test_pac_allorhythmias():
    assert [f.code for f in pac_findings([2, 4, 6, 8], 12)] == ["atrial_bigeminy"]
    assert [f.code for f in pac_findings([2, 5, 8, 11], 13)] == ["atrial_trigeminy"]
    assert [f.code for f in pac_findings([5], 12)] == ["supraventricular_ectopy"]


def test_ventricular_trigeminy():
    assert [f.code for f in pvc_findings([2, 5, 8, 11], 13)] == ["ventricular_trigeminy"]


def test_ventricular_bigeminy_is_found_when_half_the_beats_are_early():
    rr = [520, 1080] * 6
    odd = {k for k in range(1, 13) if k % 2 == 1}
    sig, rp = _train(rr, odd=odd)
    pvc = detect_pvc_beats(sig, rp, 500.0)
    assert len(pvc) >= 4
    assert [f.code for f in pvc_findings(pvc, len(rp))] == ["ventricular_bigeminy"]


def _m(rr_ms, pr_std=10.0):
    rp = np.cumsum([200] + [int(r / 2) for r in rr_ms])        # 500 Hz
    return Measurements(r_peaks_samples=[int(v) for v in rp], pr_std_ms=pr_std, p_onset_ratio=0.9,
                        intervals=Intervals(pr_ms=160))


def test_sinus_arrhythmia_needs_a_wide_rr_range_and_a_stable_pr():
    rr = [700, 760, 840, 920, 980, 920, 840, 760, 700, 760]
    assert sinus_arrhythmia_finding(_m(rr), 500.0).code == "sinus_arrhythmia"
    assert sinus_arrhythmia_finding(_m([800, 820, 790, 810, 800, 830]), 500.0) is None
    assert sinus_arrhythmia_finding(_m(rr, pr_std=60.0), 500.0) is None


def test_bifascicular_block_with_a_long_pr_is_named_trifascicular():
    from test_fascicular import LAFB, LAFB_SHAPE, _find
    from test_fascicular import _m as fm

    m = fm(-60, qrs=140, rs=LAFB_SHAPE, q={"aVL"}, v1_terminal=True)
    m.intervals.pr_ms = 260
    m.p_onset_ratio, m.pr_std_ms, m.p_wave_ratio = 0.9, 8.0, 0.9
    f = next(x for x in _find(m).findings if x.code == LAFB)
    assert "трёхпучковая" in f.label_ru
