"""D282 (electrical alternans) and D284 (ectopic atrial rhythm): synthetic positives, plain negatives."""
from __future__ import annotations

import numpy as np
import pytest

from cardiolens.arrhythmia.alternans import ALT_LEADS_MIN, detect_alternans
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline


def _alternate(sig, rpeaks, k):
    x = sig.signal.astype(float).copy()
    h = int(0.07 * sig.sampling_rate_hz)
    for i, r in enumerate(rpeaks):
        if i % 2:
            x[:, max(0, r - h):r + h] *= 1 - k
    sig.signal = x.astype(np.float32)
    return sig


@pytest.mark.parametrize("hr", [75, 110])
def test_alternans_of_20_percent_is_found(hr):
    rp = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=hr, seed=1), mode="A").measurements.r_peaks_samples
    res = run_pipeline(_alternate(generate_synthetic_ecg(heart_rate_bpm=hr, seed=1), rp, 0.2), mode="A")
    f = res.findings.by_code("electrical_alternans")
    assert f is not None and not f.critical and "перикард" in f.label_ru


def test_no_alternans_on_a_plain_record():
    res = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=110, seed=2), mode="A")
    assert "electrical_alternans" not in res.findings.codes()
    ok, det = detect_alternans(res.signal.signal, list(res.signal.lead_names), res.measurements.r_peaks_samples,
                               res.signal.sampling_rate_hz)
    assert not ok and len(det["hits"]) < ALT_LEADS_MIN


def test_alternating_rr_is_bigeminy_not_alternans():
    sig = generate_synthetic_ecg(heart_rate_bpm=75, seed=3)
    rp = np.asarray(run_pipeline(sig, mode="A").measurements.r_peaks_samples)
    # every other beat 25 % early: the amplitudes may alternate, the RR does too -> not alternans
    fake = rp.copy()
    fake[1::2] = fake[1::2] - int(0.25 * np.median(np.diff(rp)))
    x = np.tile(np.sin(np.linspace(0, 1, sig.signal.shape[1])), (sig.signal.shape[0], 1)) * 0
    for i, r in enumerate(fake):
        x[:, max(0, r - 10):r + 10] = 1000.0 if i % 2 else 600.0
    ok, det = detect_alternans(x, list(sig.lead_names), list(fake), sig.sampling_rate_hz)
    assert det["rr_flip"] > 0.6 and not ok


def test_inverted_p_in_ii_and_avf_is_an_ectopic_atrial_rhythm():
    res = run_pipeline(generate_synthetic_ecg(p_amp_scale=-1.0, seed=4), mode="A")
    f = res.findings.by_code("ectopic_atrial_rhythm")
    assert f is not None and not f.critical and "электрод" in f.label_ru


def test_upright_p_is_not_ectopic_and_short_pr_is_not_wired():
    codes = run_pipeline(generate_synthetic_ecg(pr_ms=100, seed=5), mode="A").findings.codes()
    assert "ectopic_atrial_rhythm" not in codes
    assert "short_pr" not in codes          # D283 not accepted: the rule exists, the pipeline does not call it


def test_short_pr_rule_itself():
    from cardiolens.arrhythmia.descriptive import short_pr_finding

    m = run_pipeline(generate_synthetic_ecg(pr_ms=100, seed=6), mode="A").measurements
    assert short_pr_finding(m, {"sinus_rhythm"}, adult=True) is not None
    assert short_pr_finding(m, {"sinus_rhythm", "wpw_preexcitation"}, adult=True) is None
    assert short_pr_finding(m, {"sinus_rhythm"}, adult=False) is None
