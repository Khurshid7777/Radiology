"""D205: the AF second path — a JSON logistic model where the AF rule is silent; it adds
"possible AF" and takes back the names that contradict it, and never sets the AF finding."""

import json

import numpy as np

import cardiolens.pipeline as pl
from cardiolens.api.service import signal_from_case
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.ml import af as AF


def test_the_model_file_is_plain_json_with_every_feature():
    model = json.loads(AF.MODEL_PATH.read_text())
    assert model["features"] == list(AF.AF_FEATURES)
    assert len(model["coef"]) == len(model["mean"]) == len(model["scale"]) == len(AF.AF_FEATURES)
    assert 0 < model["threshold"] < 1


def test_probability_is_a_logistic_of_the_standardised_features():
    model = {"mean": [0.0] * len(AF.AF_FEATURES), "scale": [1.0] * len(AF.AF_FEATURES),
             "coef": [0.0] * len(AF.AF_FEATURES), "intercept": 0.0}
    assert AF.af_probability({}, model) == 0.5


def test_a_regular_sinus_record_stays_sinus():
    r = pl.run_pipeline(signal_from_case("normal"), mode="A")
    assert "possible_atrial_fibrillation" not in r.findings.codes() and "sinus_rhythm" in r.findings.codes()


def test_raised_possible_af_withdraws_sinus_and_ectopy(monkeypatch):
    monkeypatch.setattr(AF, "af_probability", lambda f, m: 0.99)
    sig = signal_from_case("normal")
    res = pl.run_pipeline(sig, mode="A")
    fs = FindingSet(findings=[Finding(code="sinus_rhythm", group="rhythm", label_ru="", label_uz="", confidence=0.9),
                              Finding(code="supraventricular_ectopy", group="rhythm", label_ru="", label_uz="",
                                      confidence=0.6)])
    audit: list[str] = []
    pl._af_second_path(res.signal, res.measurements, fs, audit)
    assert fs.codes() == {"possible_atrial_fibrillation"} and "atrial_fibrillation" not in fs.codes()


def test_the_rule_af_is_left_alone(monkeypatch):
    monkeypatch.setattr(AF, "af_probability", lambda f, m: 0.99)
    res = pl.run_pipeline(signal_from_case("normal"), mode="A")
    fs = FindingSet(findings=[Finding(code="atrial_fibrillation", group="rhythm", label_ru="", label_uz="",
                                      confidence=0.7)])
    pl._af_second_path(res.signal, res.measurements, fs, [])
    assert fs.codes() == {"atrial_fibrillation"}


def test_features_tolerate_a_short_strip():
    res = pl.run_pipeline(signal_from_case("normal"), mode="A")
    res.measurements.r_peaks_samples = res.measurements.r_peaks_samples[:3]
    f = AF.af_features(res.signal, res.measurements)
    assert set(f) == set(AF.AF_FEATURES_V2) and np.isfinite(list(f.values())).all()
