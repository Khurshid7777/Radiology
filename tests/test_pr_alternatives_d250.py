"""D250: two PR estimates for the scanner's PR stability check; the pipeline's PR is unchanged."""
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure, pr_alternatives


def test_both_estimates_on_a_clean_sinus_record():
    alt = pr_alternatives(generate_synthetic_ecg(pr_ms=180.0))
    assert set(alt) == {"pr_single_ms", "pr_multi_ms", "delta_ms", "n_beats"}
    assert alt["pr_single_ms"] is not None and alt["pr_multi_ms"] is not None
    assert abs(alt["delta_ms"]) <= 20 and alt["n_beats"] >= 5


def test_measurements_input_and_pr_ms_untouched():
    sig = generate_synthetic_ecg(pr_ms=200.0)
    m = measure(sig)
    alt = pr_alternatives(m)
    assert alt["pr_single_ms"] == m.intervals.pr_ms


def test_delta_is_none_without_a_second_estimate():
    m = Measurements(intervals=Intervals(pr_ms=160.0))
    assert pr_alternatives(m) == {"pr_single_ms": 160.0, "pr_multi_ms": None, "delta_ms": None, "n_beats": None}
