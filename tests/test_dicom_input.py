"""D94: DICOM waveform ECG input (12-lead / general / ambulatory ECG storage)."""

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from cardiolens.layer1_ingestion import generate_synthetic_ecg, is_dicom, load, load_dicom
from dicom_builder import group as _group  # tests/dicom_builder.py
from dicom_builder import write as _write


@pytest.fixture(scope="module")
def ecg():
    return generate_synthetic_ecg(heart_rate_bpm=72, seed=3)


def test_twelve_leads_read_in_microvolts(tmp_path, ecg):
    f = tmp_path / "ecg.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)])
    sig = load_dicom(f)
    assert sig.source == "dicom" and sig.lead_names == ecg.lead_names
    assert sig.sampling_rate_hz == ecg.sampling_rate_hz and sig.signal.shape == ecg.signal.shape
    assert np.abs(sig.signal - ecg.signal).max() <= 1.25 + 1e-3            # half a 2.5-uV count
    assert sig.patient.age_years == 45 and sig.patient.sex == "female"
    assert sig.provenance.digitized is False


def test_units_baseline_and_correction_factor(tmp_path, ecg):
    f = tmp_path / "mv.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz, unit="mV", sens=0.005,
                      baseline=-12.0, correction=0.98)])
    sig = load_dicom(f)
    assert np.abs(sig.signal - ecg.signal).max() <= 0.5 * 5 * 0.98 + 1e-3


def test_leads_named_by_code_only(tmp_path, ecg):
    f = tmp_path / "codes.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz, by_code=True)])
    assert load_dicom(f).lead_names == ecg.lead_names


def test_rhythm_group_preferred_over_median_beat(tmp_path, ecg):
    f = tmp_path / "two.dcm"
    median = _group(ecg.signal[:, :600], ecg.lead_names, ecg.sampling_rate_hz, label="MEDIAN BEAT")
    _write(f, [median, _group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)])
    assert load_dicom(f).signal.shape[1] == ecg.signal.shape[1]


def test_identity_never_leaves_the_file(tmp_path, ecg):
    f = tmp_path / "pii.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)])
    dumped = load_dicom(f).model_dump_json(exclude={"signal"})
    assert "Ivanov" not in dumped and "MRN-12345" not in dumped and "19800101" not in dumped


def test_not_an_ecg_or_no_scale_is_refused(tmp_path, ecg):
    f = tmp_path / "ct.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)], sop="1.2.840.10008.5.1.4.1.1.2")
    with pytest.raises(ValueError, match="not an ECG"):
        load_dicom(f)
    g = _group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)
    del g.ChannelDefinitionSequence[0].ChannelSensitivity
    f2 = tmp_path / "noscale.dcm"
    _write(f2, [g])
    with pytest.raises(ValueError, match="no physical scale"):
        load_dicom(f2)


def test_dispatch_by_extension_and_magic(tmp_path, ecg):
    f = tmp_path / "ecg.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)])
    assert load(f).source == "dicom" and is_dicom(f.read_bytes())
    assert not is_dicom(b"x" * 200)


def test_api_accepts_a_device_export_without_extension(tmp_path, ecg):
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    f = tmp_path / "ecg.dcm"
    _write(f, [_group(ecg.signal, ecg.lead_names, ecg.sampling_rate_hz)])
    r = TestClient(app, headers={"X-API-Key": "test-key"}).post("/api/analyze", files=[("files", ("00000017", f.read_bytes(), "application/octet-stream"))])
    assert r.status_code == 200, r.text
    hr = r.json()["measurements"]["heart_rate_bpm"]
    assert hr == pytest.approx(72, abs=3)
