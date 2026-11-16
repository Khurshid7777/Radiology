"""D292: P waves at twice the ventricular rate in a regular bradycardia -> probable 2 : 1 AV block, not sinus
bradycardia; a plain sinus bradycardia and a dissociated (complete) block are not called 2 : 1."""
import pytest

from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline


@pytest.mark.parametrize("hr", [40, 45])
def test_two_p_per_qrs_is_a_probable_2to1_block(hr):
    res = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=hr, dissociated_atrial_rate_bpm=2 * hr, seed=1), mode="A")
    codes = res.findings.codes()
    f = res.findings.by_code("probable_av_block_2to1")
    assert f is not None and not f.critical and "2 : 1" in f.label_ru
    assert "sinus_bradycardia" not in codes


def test_plain_sinus_bradycardia_is_not_2to1():
    codes = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=40, seed=2), mode="A").findings.codes()
    assert "probable_av_block_2to1" not in codes and "sinus_bradycardia" in codes


def test_dissociated_atria_stay_a_complete_block():
    codes = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=40, dissociated_atrial_rate_bpm=75, seed=3),
                         mode="A").findings.codes()
    assert "probable_av_block_2to1" not in codes


def test_descriptive_findings_are_not_flagged_implausible():
    # Pre-deploy review 16.11: a finding no causal rule explains read as "implausible — artefact / swapped electrodes".
    for kw in ({"heart_rate_bpm": 40, "dissociated_atrial_rate_bpm": 80}, {"p_amp_scale": -1.0}):
        res = run_pipeline(generate_synthetic_ecg(seed=1, **kw), mode="A")
        assert not set(res.causal_graph.implausible_flags) & {"probable_av_block_2to1", "ectopic_atrial_rhythm"}


def test_syncope_with_a_2to1_block_names_the_block():
    from cardiolens.clinical_context import advise

    out = advise(["syncope"], None, {"probable_av_block_2to1"},
                 {"probable_av_block_2to1": "Вероятна АВ-блокада 2 : 1"})
    assert out and out[0].urgent and "АВ-блокада 2 : 1" in out[0].ru
