"""SD-166: the pure parts of the culprit-lead diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("qrs_culprit", Path(__file__).parents[2] / "tools" / "scanner" / "qrs_culprit.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


def test_the_truth_is_put_on_the_reads_time_axis():
    x = np.arange(10.0)
    s, start = qc.shifted_truth(x, 2.5, 0.1, 10.0)          # read time t = truth time t + lag: the truth starts 0.1 s earlier
    assert start == 2.4 and s is x
    s, start = qc.shifted_truth(x, 0.0, 0.2, 10.0)          # would start before 0: the first samples are cut
    assert start == 0.0 and list(s) == list(x[2:])


def test_a_culprit_is_a_lead_whose_truth_brings_the_error_under_the_line():
    assert qc.culprits({"II:rhythm": 8.0, "V1:panel": 40.0, "I:panel": None}, 94.0, 20.0) == ["II:rhythm"]
    assert qc.culprits({"II:rhythm": 8.0}, 10.0, 20.0) == []               # nothing to explain
    assert qc.culprits({"II:rhythm": 8.0}, None, 20.0) == []


def test_where_the_read_differs_most_from_the_truth():
    truth, read = np.zeros(100), np.zeros(100)
    read[55] = 1.0
    assert qc.where_worst(truth, read, np.array([20, 50, 80]), 100.0) == 0.05
    assert qc.where_worst(truth, read, np.array([], int), 100.0) is None
