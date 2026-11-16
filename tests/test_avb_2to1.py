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
