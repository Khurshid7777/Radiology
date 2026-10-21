"""ScanResult: pydantic model and JSON Schema agree; the honesty rules hold (TZ §7)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from ecgscan.contract import SCHEMA_PATH, Calibration, ScanResult, Trace

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_shipped_example_is_valid_for_both(validator):
    ex = json.loads(SCHEMA_PATH.with_name("scan_result.example.json").read_text(encoding="utf-8"))
    assert list(validator.iter_errors(ex)) == []
    assert ScanResult.model_validate(ex).status == "needs_confirmation"


def test_model_output_validates_against_schema(validator, result_factory):
    tr = Trace.from_array(np.array([0.1, np.nan, 0.3]), start_s=0.0, fs=500, lead="II", quality=0.5)
    res = result_factory(rhythm=[tr], warnings=["TRACE_GAPS"])
    d = res.to_json_dict()
    assert d["rhythm"][0]["samples_mv"] == [0.1, None, 0.3]      # missing stays missing
    assert list(validator.iter_errors(d)) == []


def test_ok_requires_established_speed_and_gain(result_factory):
    with pytest.raises(ValidationError):
        result_factory(calibration=Calibration(paper_speed_mm_s=None, gain_mm_mv=10, confidence=0.3))
    res = result_factory(status="needs_confirmation",
                         calibration=Calibration(paper_speed_mm_s=None, gain_mm_mv=10, confidence=0.3))
    assert res.calibration.paper_speed_mm_s is None


def test_rejected_iff_reasons(result_factory):
    with pytest.raises(ValidationError):
        result_factory(status="rejected")
    with pytest.raises(ValidationError):
        result_factory(reject_reasons=["BLUR"])
    assert result_factory(status="rejected", reject_reasons=["LOW_RESOLUTION"]).reject_reasons == ["LOW_RESOLUTION"]


def test_unknown_codes_speeds_and_leads_are_refused(result_factory):
    with pytest.raises(ValidationError):
        result_factory(status="rejected", reject_reasons=["NOT_A_CODE"])
    with pytest.raises(ValidationError):
        result_factory(calibration=Calibration(paper_speed_mm_s=30, gain_mm_mv=10, confidence=1.0))
    with pytest.raises(ValidationError):
        result_factory(leads={"MLII": Trace(samples_mv=[0.0], start_s=0, duration_s=0.002, quality=1)})


def test_digitized_flag_is_mandatory(result_factory):
    with pytest.raises(ValidationError):
        result_factory(provenance_flags=[])


def test_trace_roundtrip_keeps_nan():
    tr = Trace.from_array(np.array([1.0, np.nan, np.nan, 2.0]), start_s=1.0, fs=4.0, quality=1.0)
    assert tr.duration_s == 1.0 and tr.missing_fraction == 0.5
    np.testing.assert_array_equal(np.isnan(tr.array()), [False, True, True, False])
