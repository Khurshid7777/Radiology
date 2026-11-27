"""Measurement robustness lessons from real PhysioNet data (TZ §6, §16).

Envelope-based QRS bounds (BBB no longer reads as narrow), amplitude-refined P
onsets (PR tracks the stimulus within a few ms) and the resulting findings, all
through the full pipeline on synthetic stimuli.
"""

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline
from cardiolens.validation import build_synthetic_dataset, run_validation


def _run(**kw):
    return run_pipeline(generate_synthetic_ecg(**kw), mode="A", provider=StubProvider())


@pytest.mark.parametrize("bbb,expect,other", [("rbbb", "rbbb", "lbbb"), ("lbbb", "lbbb", "rbbb")])
def test_bundle_branch_block_end_to_end(bbb, expect, other):
    r = _run(heart_rate_bpm=70, bbb=bbb, seed=6)
    assert r.measurements.intervals.qrs_ms >= 120
    codes = r.findings.codes()
    assert expect in codes and other not in codes
    assert "wide_qrs_nonspecific" not in codes


def test_normal_qrs_is_narrow_and_not_a_block():
    r = _run(heart_rate_bpm=75, seed=9)
    assert 60 <= r.measurements.intervals.qrs_ms <= 110
    assert not r.findings.codes() & {"rbbb", "lbbb", "wide_qrs_nonspecific", "wpw_preexcitation"}


@pytest.mark.parametrize("pr", [150, 240, 280, 300])
def test_pr_tracks_stimulus(pr):
    r = _run(heart_rate_bpm=70, pr_ms=pr, seed=21)
    # D290: the P onset is calibrated to the LUDB annotators' (they mark it ~10 ms before the generator's onset), so
    # on the synthetic stimulus the PR reads that much longer; the 200-ms boundary below still holds.
    from cardiolens.layer3_measure.measure import PR_ONSET_OFFSET_MS
    assert r.measurements.intervals.pr_ms == pytest.approx(pr + PR_ONSET_OFFSET_MS, abs=12)
    assert ("first_degree_av_block" in r.findings.codes()) == (pr > 200)


def test_synthetic_interval_mae_is_small():
    rep = run_validation(build_synthetic_dataset())
    assert rep.pr_mae_ms is not None and rep.pr_mae_ms <= 15
    assert rep.qt_mae_ms is not None and rep.qt_mae_ms <= 15
    assert rep.macro_sensitivity == 1.0


def test_slow_rate_does_not_double_count_t_waves():
    r = _run(heart_rate_bpm=40, pr_ms=160, seed=20)
    assert r.measurements.heart_rate_bpm == pytest.approx(40, abs=2)
    assert "sinus_bradycardia" in r.findings.codes()


def test_terminal_sign_ignores_discordant_st_segment():
    """LBBB-like V1 (rS with elevated ST after the QRS): the terminal forces must be
    read inside the lead's own QRS, not in the ST segment that the multi-lead
    envelope offset can include (D41)."""
    import numpy as np

    from cardiolens.layer3_measure.delineation import terminal_qrs_sign

    fs = 500.0
    n = int(2.0 * fs)
    row = np.zeros(n)
    r = 500
    # small r (10 ms), deep broad S (60 ms) returning to a +300 µV elevated ST plateau
    row[r - 5: r] = 200
    s_len = int(0.06 * fs)
    row[r: r + s_len] = -1500 * np.sin(np.linspace(0, np.pi, s_len))
    row[r + s_len: r + s_len + int(0.20 * fs)] = 300           # discordant ST elevation
    late_off = np.array([r + s_len + int(0.03 * fs)], dtype=float)   # envelope offset 30 ms into the ST
    on = np.array([r - 5], dtype=float)
    assert terminal_qrs_sign(row, np.array([r]), late_off, fs, on) is False
    # RBBB-like V1: rSR' — terminal forces positive even with the same late offset.
    row2 = np.zeros(n)
    row2[r - 5: r] = 300
    row2[r: r + 20] = -400
    row2[r + 20: r + 20 + 40] = 900 * np.sin(np.linspace(0, np.pi, 40))       # broad R'
    late_off2 = np.array([r + 60 + int(0.03 * fs)], dtype=float)
    assert terminal_qrs_sign(row2, np.array([r]), late_off2, fs, on) is True
