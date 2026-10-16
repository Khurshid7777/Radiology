"""Adult rhythm findings ported from claude/vibrant-curie-ro1zsz (D25, D37, D38, D40, D41):
ventricular runs (VT, §10), 12-lead PVCs / bigeminy, pauses, the signal-organization gate,
and the SVT / wide-complex-tachycardia split at 140/min."""

import numpy as np

from cardiolens.arrhythmia import (
    detect_pvc_beats,
    detect_ventricular_runs,
    pvc_findings,
    vt_run_finding,
)
from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.findings import FindingSet
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.pipeline import run_pipeline
from cardiolens.safety import scan_critical


def _pulse_train(rr_ms, odd=None, fs=500.0):
    """3-lead (II, V1, V5) train of narrow QRS-like pulses at the given RR intervals;
    beats listed in ``odd`` get a wide biphasic shape instead."""
    t_r = np.cumsum([400.0] + list(rr_ms)) / 1000.0
    n = int((t_r[-1] + 0.5) * fs)
    x = np.zeros((3, n))
    tt = np.arange(n) / fs
    for k, t in enumerate(t_r):
        if odd and k in odd:
            pulse = 900 * (np.exp(-((tt - t + 0.03) / 0.03) ** 2) - np.exp(-((tt - t - 0.04) / 0.03) ** 2))
        else:
            pulse = 1000 * np.exp(-((tt - t) / 0.012) ** 2)
        x += pulse
    sig = SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=fs,
                               lead_names=["II", "V1", "V5"], duration_s=n / fs, source="synthetic")
    return sig, (t_r * fs).astype(int)


def test_ventricular_run_detector():
    rr = [800] * 5 + [500, 480, 480, 1100] + [800] * 4
    sig, rp = _pulse_train(rr, odd={6, 7, 8})
    runs = detect_ventricular_runs(sig, rp, 500.0)
    assert runs == [(6, 3, 125)]
    f = vt_run_finding(runs)
    assert f.code == "ventricular_tachycardia" and f.critical
    assert any(a.code == "ventricular_tachycardia"
               for a in scan_critical(FindingSet(findings=[f]), Measurements()).alerts)
    # A couplet is not a run; a slow (idioventricular) run is not VT; same-shaped beats are not V.
    sig, rp = _pulse_train([800] * 5 + [500, 480, 1100] + [800] * 5, odd={6, 7})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    sig, rp = _pulse_train([800] * 5 + [700, 700, 700, 1100] + [800] * 4, odd={6, 7, 8})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    sig, rp = _pulse_train(rr)
    assert detect_ventricular_runs(sig, rp, 500.0) == []


def test_pvc_rule_and_bigeminy():
    rr = [800] * 5 + [560, 1040] + [800] * 5
    sig, rp = _pulse_train(rr, odd={6})
    assert detect_pvc_beats(sig, rp, 500.0) == [6]
    sig, rp = _pulse_train(rr)                     # premature but same shape (PAC)
    assert detect_pvc_beats(sig, rp, 500.0) == []
    assert [f.code for f in pvc_findings([2, 4, 6, 8], 12)] == ["ventricular_bigeminy"]
    assert [f.code for f in pvc_findings([5], 12)] == ["ventricular_ectopy"]


def test_pause_measured_flagged_and_critical():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    fs = int(sig.sampling_rate_hz)
    a = sig.signal.shape[1] // 2
    sig.signal[:, a:a + int(1.6 * fs)] = 0.0
    res = run_pipeline(sig, mode="A")
    assert res.measurements.max_rr_ms >= 1500 and res.measurements.pause_ratio >= 1.8
    assert "sinus_pause" in res.findings.codes()
    crit = scan_critical(FindingSet(), Measurements(max_rr_ms=3200))
    assert any(al.code == "ventricular_pause" for al in crit.alerts)


def test_disorganized_signal_withholds_findings():
    rng = np.random.RandomState(0)
    noise = SignalRepresentation(
        signal=rng.normal(0, 300, (12, 5000)).astype("float32"), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=10, source="synthetic",
        patient=Patient(age_years=50, sex="male"))
    codes = run_pipeline(noise, mode="A").findings.codes()
    assert "poor_quality_signal" in codes
    assert not ({"sinus_bradycardia", "sinus_rhythm", "sinus_tachycardia"} & codes)
    clean = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.codes()
    assert "poor_quality_signal" not in clean


def _find(m):
    sig = SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500,
                               lead_names=list(STANDARD_12_LEADS), duration_s=0.02,
                               source="synthetic", patient=Patient(age_years=50, sex="male"))
    return detect_findings(sig, m, norms=load_norms("adult"))


def test_wide_tachycardia_is_not_sinus_and_narrow_is_svt():
    m = Measurements(heart_rate_bpm=175, rhythm_regular=True, qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=100, qrs_ms=140, qt_ms=300, qtc_bazett_ms=510))
    fs = _find(m)
    assert "wide_complex_tachycardia" in fs.codes()
    assert not {"sinus_tachycardia", "wpw_preexcitation", "first_degree_av_block"} & fs.codes()
    assert [a.code for a in scan_critical(fs, m).alerts].count("wide_complex_tachycardia") == 1
    narrow = Measurements(heart_rate_bpm=175, rhythm_regular=True,
                          intervals=Intervals(pr_ms=120, qrs_ms=90, qt_ms=280, qtc_bazett_ms=470))
    assert "supraventricular_tachycardia" in _find(narrow).codes()
    slow = Measurements(heart_rate_bpm=120, rhythm_regular=True,
                        intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=320, qtc_bazett_ms=450))
    assert "sinus_tachycardia" in _find(slow).codes()
    # Backstop alert from 140/min when Layer 4 did not emit the finding.
    assert any(a.code == "wide_complex_tachycardia" for a in
               scan_critical(FindingSet(), Measurements(heart_rate_bpm=145,
                                                        intervals=Intervals(qrs_ms=150))).alerts)


# --- Paced rhythm (source D24, D74, D75) ---
from cardiolens.layer3_measure.pacing import (
    pacing_multilead_spike_times,
    pacing_spike_coupling,
    pacing_spike_ratio,
)


def _r_times(sig):
    return [r / sig.sampling_rate_hz for r in
            run_pipeline(sig.model_copy(deep=True), mode="A").measurements.r_peaks_samples]


def _spiked_ecg(offset_s, leads=None, scale=3.0):
    base = generate_synthetic_ecg(heart_rate_bpm=72)
    r_s = _r_times(base)
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    fs = sig.sampling_rate_hz
    for name in (leads or ["II"]):
        li = sig.lead_names.index(name)
        amp = float(np.max(np.abs(sig.signal[li]))) * scale
        for t in r_s:
            i = int(round((t + offset_s) * fs))
            if 0 <= i < sig.signal.shape[1]:
                sig.signal[li, i] += amp
    return sig


def test_pacing_spike_coupling_math():
    r = [1.0, 2.0, 3.0, 4.0]
    assert pacing_spike_coupling([0.95, 1.95, 2.95, 3.95], r) == 1.0
    assert pacing_spike_coupling([1.5, 2.5], r) == 0.0
    assert pacing_spike_coupling([0.99], r) == 0.25


def test_paced_call_requires_spike_locked_to_qrs():
    assert pacing_spike_ratio(generate_synthetic_ecg(heart_rate_bpm=72)) < 5.5
    locked = run_pipeline(_spiked_ecg(-0.04), mode="A").findings.codes()
    assert "paced_rhythm" in locked and "sinus_rhythm" not in locked
    assert "paced_rhythm" not in run_pipeline(_spiked_ecg(+0.40), mode="A").findings.codes()


def test_af_is_withheld_on_a_paced_rhythm(monkeypatch):
    # D91: paced beats have no P and the pacer sets the RR, so the P/RR AF rule
    # does not apply under a pacemaker.
    import cardiolens.pipeline as P
    from cardiolens.contracts.findings import Finding

    real = P.detect_findings

    def with_af(*a, **k):
        fs = real(*a, **k)
        fs.findings.append(Finding(code="atrial_fibrillation", group="rhythm", label_ru="ФП",
                                   label_uz="FP", confidence=0.7))
        return fs

    monkeypatch.setattr(P, "detect_findings", with_af)
    paced = run_pipeline(_spiked_ecg(-0.04), mode="A").findings.codes()
    assert "paced_rhythm" in paced and "atrial_fibrillation" not in paced
    assert "atrial_fibrillation" in run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.codes()


def test_multilead_spike_detects_small_qrs_locked_spikes():
    sig = _spiked_ecg(-0.04, leads=list(STANDARD_12_LEADS), scale=0.6)
    assert len(pacing_multilead_spike_times(sig)) >= 5
    one = _spiked_ecg(-0.04, leads=["II"], scale=0.6)
    assert pacing_multilead_spike_times(one) == []


# --- Pathological Q and Brugada type 1 (source D36, D89, D90) ---
from cardiolens.contracts.features import Morphology

_NO_Q = (0.0, 0.0, 800.0, False)


def _q_codes(q_waves, qrs_ms=95, v1_term=None, norms="adult", age_years=60):
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=v1_term,
                     intervals=Intervals(pr_ms=160, qrs_ms=qrs_ms, qt_ms=390, qtc_bazett_ms=420),
                     morphology=Morphology(q_waves=q_waves))
    sig = SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500,
                               lead_names=list(STANDARD_12_LEADS), duration_s=0.02,
                               source="synthetic", patient=Patient(age_years=age_years, sex="male"))
    return detect_findings(sig, m, norms=load_norms(norms))


def test_pathological_q_contiguous_leads_and_depth():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["III"] = (44.0, 300.0, 400.0, False)
    assert "pathological_q_waves" not in _q_codes(q).codes()       # isolated Q in III
    q["aVF"] = (40.0, 250.0, 500.0, False)
    f = _q_codes(q).by_code("pathological_q_waves")
    assert f is not None and "III,aVF" in f.label_ru and "нижняя" in f.label_ru
    q["aVF"] = (40.0, 60.0, 900.0, False)                           # shallow
    assert "pathological_q_waves" not in _q_codes(q).codes()


def test_pathological_q_v2_v3_and_absent_in_children():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["V2"] = q["V3"] = (0.0, 0.0, 20.0, True)
    assert "pathological_q_waves" in _q_codes(q).codes()
    assert "pathological_q_waves" not in _q_codes(q, norms="pediatric", age_years=8).codes()


def _brugada_m(v1=(450.0, 300.0, 150.0, -250.0), s_v6=-200.0):
    m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=110))
    m.morphology.right_precordial_st = {"V1": v1, "V2": (120.0, 100.0, 80.0, -60.0)}
    m.morphology.terminal_s_lateral = {"I": -150.0, "V6": s_v6}
    return m


def test_brugada_type1_coved_pattern():
    f = _find(_brugada_m()).by_code("brugada_pattern")
    assert f is not None and "V1" in f.label_ru and f.confidence == 0.3
    assert "brugada_pattern" not in _find(_brugada_m(v1=(450.0, 480.0, 500.0, -250.0))).codes()  # saddleback
    assert "brugada_pattern" not in _find(_brugada_m(v1=(450.0, 300.0, 150.0, 80.0))).codes()    # upright T
    assert "brugada_pattern" not in _find(_brugada_m(s_v6=-400.0)).codes()                        # true RBBB


# --- R-peak recovery from the multi-lead envelope (D75) ---
def test_recover_missed_qrs_fills_gaps_but_not_spikes():
    from cardiolens.layer3_measure.measure import _recover_missed_qrs

    fs = 500.0
    tt = np.arange(5000) / fs
    beats = np.arange(0.3, 9.8, 0.6)                      # 100/min
    x = np.zeros((12, tt.size))
    for t in beats:
        x += 1000 * np.exp(-((tt - t) / 0.012) ** 2)        # sharp QRS in every lead
        x += 150 * np.exp(-((tt - t - 0.3) / 0.06) ** 2)    # broad T
    idx = (beats * fs).astype(int)
    # The single-lead detector kept an irregular subset (PTB-XL 18191: 4 of 12 beats).
    kept = idx[[0, 1, 4, 5, 9, 11, 12, 15]]
    out, n_rec = _recover_missed_qrs(x, kept, fs)
    assert n_rec == len(idx) - len(kept)
    assert np.all(np.abs(np.sort(out) - np.sort(idx)) <= 10)   # envelope peak within 20 ms of R
    # A complete list gets nothing added; a spike inside every RR (pacemaker-like)
    # would make the rhythm less regular and is refused.
    assert _recover_missed_qrs(x, idx, fs)[1] == 0
    y = x.copy()
    for t in beats[:-1]:
        y[:, int((t + 0.15) * fs)] += 3000
    assert _recover_missed_qrs(y, idx, fs)[1] == 0


def test_recover_missed_qrs_skips_a_noisy_strip():
    # Noise between beats at half the beats' level (PTB-XL 15973): recovery would count
    # noise peaks as beats, so it is skipped and says why (D79).
    from cardiolens.layer3_measure.measure import _recover_missed_qrs

    fs = 500.0
    tt = np.arange(5000) / fs
    beats = np.arange(0.3, 9.8, 0.6)
    x = np.zeros((12, tt.size))
    for t in beats:
        x += 1000 * np.exp(-((tt - t) / 0.012) ** 2)
    x += 120 * np.random.default_rng(0).standard_normal(x.shape)
    idx = (beats * fs).astype(int)
    notes: list[str] = []
    out, n_rec = _recover_missed_qrs(x, idx[[0, 1, 4, 5, 9, 11, 12, 15]], fs, notes)
    assert n_rec == 0 and len(out) == 8
    assert notes and "noisy envelope" in notes[0]


def test_recover_missed_qrs_takes_the_whole_wide_complex_as_reference():
    # LBBB-like QRS: a weak first envelope hump (where the detector sits) and a dominant
    # second hump 130 ms later; T waves 270 ms after that. Half the first hump's level let
    # the T waves through (PTB-XL 13820: 124/min at 75); the whole complex does not (D80).
    from cardiolens.layer3_measure.measure import _recover_missed_qrs

    fs = 500.0
    tt = np.arange(5000) / fs
    beats = np.arange(0.4, 9.6, 0.8)                        # 75/min
    x = np.zeros((12, tt.size))
    for t in beats:
        x += 400 * np.exp(-((tt - t) / 0.012) ** 2)
        x -= 900 * np.exp(-((tt - t - 0.13) / 0.014) ** 2)
        x += 560 * np.exp(-((tt - t - 0.40) / 0.035) ** 2)
    idx = (beats * fs).astype(int)
    out, n_rec = _recover_missed_qrs(x, idx[[0, 1, 4, 5, 6, 8, 10, 11]], fs)
    assert n_rec == 4 and len(out) == len(beats)


def test_recovered_beats_marked_off_their_r_peak_do_not_make_a_vt_run():
    # Envelope-recovered beats (D75) sit at the envelope peak, not the rhythm lead's R.
    # Three normal beats marked 40 ms early read as an abnormally shaped premature run
    # unless they are compared at their best alignment (PTB-XL 15108, 4885; D87).
    from cardiolens.arrhythmia.beats import detect_ventricular_runs
    from cardiolens.layer6_interpret import StubProvider

    sig = generate_synthetic_ecg(heart_rate_bpm=150, seed=3)
    rp = list(run_pipeline(sig.model_copy(deep=True), mode="A", provider=StubProvider())
              .measurements.r_peaks_samples)
    fs = sig.sampling_rate_hz
    off = int(-0.040 * fs)
    moved = set(rp[5:8])
    shifted = [r + off if r in moved else r for r in rp]
    assert not detect_ventricular_runs(sig, shifted, fs, recovered=[r + off for r in moved])
    # A detector beat displaced the same way is rescued too when its shifted match is
    # near-perfect (PTB-XL 18619: 0.998; true PVCs there 0.33-0.81; D88) ...
    assert not detect_ventricular_runs(sig, shifted, fs)
    # ... and without that rescue the displaced beats are the false run it removes.
    from cardiolens.arrhythmia import beats
    keep = beats._RUN_SHIFTED_RESCUE_MIN
    try:
        beats._RUN_SHIFTED_RESCUE_MIN = 2.0
        assert detect_ventricular_runs(sig, shifted, fs)
    finally:
        beats._RUN_SHIFTED_RESCUE_MIN = keep


# --- QRS offset past a terminal R' after an envelope dip (D90) ---

def test_qrs_offset_continues_past_a_terminal_r_prime():
    import cardiolens.layer3_measure.delineation as D

    fs = 500.0
    tt = np.arange(5000) / fs
    beats = np.arange(0.5, 9.5, 0.75)
    x = np.zeros((12, tt.size))
    for t in beats:
        x += 1000 * np.exp(-((tt - t) / 0.012) ** 2)            # main QRS deflection
        x += 350 * np.exp(-((tt - t - 0.085) / 0.012) ** 2)     # terminal R', after a 30 ms dip
        x += 300 * np.exp(-((tt - t - 0.33) / 0.04) ** 2)       # T wave
    r = (beats * fs).astype(int)
    on, off = D.qrs_bounds_envelope(x, r, fs)
    assert np.nanmedian((off - on) / fs * 1000) >= 120
    old = D.QRS_HUMP_FRACTION
    try:
        D.QRS_HUMP_FRACTION = 0.0
        on0, off0 = D.qrs_bounds_envelope(x, r, fs)
    finally:
        D.QRS_HUMP_FRACTION = old
    assert np.nanmedian((off0 - on0) / fs * 1000) < 100


def test_wave_marks_missing_an_edge_beat_stay_on_their_own_beats():
    from cardiolens.layer3_measure.measure import _aligned

    r = np.array([10, 721, 1187, 1897])
    p_on = _aligned([584, 1100, 1815], r, before=True)       # no P onset for the edge beat
    assert np.isnan(p_on[0]) and list(p_on[1:]) == [584, 1100, 1815]
    t_off = _aligned([300, 1000, 1450], r, before=False)     # no T offset for the last beat
    assert list(t_off[:3]) == [300, 1000, 1450] and np.isnan(t_off[3])
    assert list(_aligned([1, 2, 3, 4], r, before=True)) == [1, 2, 3, 4]
