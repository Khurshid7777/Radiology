"""D97: a record that cannot be measured is refused with the reason, never measured on guesses."""

import numpy as np
import pytest

from cardiolens.config import IngestionConfig
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg, load_csv, load_edf, load_scp
from cardiolens.layer1_ingestion.checks import check_record
from cardiolens.pipeline import run_pipeline

LIMITS = IngestionConfig()


@pytest.fixture(scope="module")
def ecg():
    return generate_synthetic_ecg(heart_rate_bpm=70, seed=4)


def _with(sig, signal, fs=None):
    fs = fs or sig.sampling_rate_hz
    return SignalRepresentation(signal=signal, sampling_rate_hz=fs, lead_names=sig.lead_names,
                                duration_s=signal.shape[1] / fs, source="csv")


# --- invalid samples ----------------------------------------------------------------

def test_a_short_nan_run_is_bridged_and_masked_and_measured(ecg):
    x = ecg.signal.copy()
    x[1, 2000:2040] = np.nan                                      # 80 ms in lead II
    res = run_pipeline(_with(ecg, x), mode="A")
    assert res.measurements.heart_rate_bpm == pytest.approx(70, abs=3)
    assert res.measurements.intervals.qrs_ms is not None
    assert res.signal.provenance.sample_mask["II"] == [(4.0, 4.08)]


def test_nan_at_the_ends_is_trimmed(ecg):
    x = ecg.signal.copy()
    x[:, -500:] = np.nan                                          # a padded last second
    out = check_record(_with(ecg, x), LIMITS)
    assert out.n_samples == x.shape[1] - 500 and np.isfinite(out.signal).all()
    assert any("trimmed" in n for n in out.provenance.notes)


@pytest.mark.parametrize(("where", "msg"), [
    ((slice(1, 2), slice(2000, 2500)), "cannot be bridged"),    # 1 s inside lead II
    ((slice(3, 4), slice(None)), "lead aVR has no valid samples"),
    ((slice(None), slice(None)), "no valid samples"),
])
def test_long_or_total_invalid_runs_are_refused(ecg, where, msg):
    x = ecg.signal.copy()
    x[where] = np.inf
    with pytest.raises(ValueError, match=msg):
        check_record(_with(ecg, x), LIMITS)


# --- rate and length ----------------------------------------------------------------

def test_too_low_a_rate_or_too_short_a_record_is_refused(ecg):
    with pytest.raises(ValueError, match="below 100 Hz"):
        run_pipeline(_with(ecg, ecg.signal[:, ::10], fs=50.0), mode="A")
    with pytest.raises(ValueError, match="at least 2 s"):
        check_record(_with(ecg, ecg.signal[:, :750]), LIMITS)
    noted = check_record(_with(ecg, ecg.signal[:, ::4], fs=125.0), LIMITS)
    assert any("coarse" in n for n in noted.provenance.notes)


# --- noise ------------------------------------------------------------------------

def test_noise_with_too_few_beats_to_judge_is_poor_quality_and_alerts_say_so(ecg):
    """Band-limited noise with 3 'beats' read as AF and HR 32, mode A, no escalation."""
    noise = np.random.default_rng(2).normal(0, 200, (12, 5000))
    res = run_pipeline(_with(ecg, noise), mode="A")
    assert [f.code for f in res.findings.findings] == ["poor_quality_signal"]
    assert res.safety.alerts and all(a.unverified for a in res.safety.alerts)       # never withheld
    assert all("низкого качества" in a.label_ru for a in res.safety.alerts)


def test_real_extreme_bradycardia_stays_verified():
    res = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=30, seed=1), mode="A")
    alert = next(a for a in res.safety.alerts if a.code == "severe_bradycardia")
    assert not alert.unverified


# --- EDF ---------------------------------------------------------------------------

def _edf(path, chans):
    pyedflib = pytest.importorskip("pyedflib")
    w = pyedflib.EdfWriter(str(path), len(chans), file_type=pyedflib.FILETYPE_EDFPLUS)
    heads = []
    for label, fs, dim, data in chans:
        lim = float(np.abs(data).max()) * 1.1 + 1
        heads.append({"label": label, "dimension": dim, "sample_frequency": fs, "physical_min": -lim,
                      "physical_max": lim, "digital_min": -32768, "digital_max": 32767})
    w.setSignalHeaders(heads)
    w.writeSamples([np.asarray(d, dtype=np.float64) for *_, d in chans])
    w.close()


def test_edf_units_per_signal_and_non_ecg_left_out(tmp_path, ecg):
    uv = ecg.signal.astype(np.float64)
    chans = [("I", 500, "uV", uv[0]), ("II", 500, "mV", uv[1] / 1000), ("V1", 500, "V", uv[6] / 1e6),
             ("Resp", 25, "Ohm", np.zeros(250))]
    _edf(tmp_path / "a.edf", chans)
    sig = load_edf(tmp_path / "a.edf")
    assert sig.lead_names == ["I", "II", "V1"] and sig.sampling_rate_hz == 500
    for k, i in enumerate((0, 1, 6)):                            # V: the 8-character EDF header
        tol = 0.1 if k == 2 else 0.02                             # rounds a sub-mV physical range
        assert np.abs(sig.signal[k] - uv[i]).max() < tol * np.abs(uv[i]).max() + 2
    assert any("Resp" in n for n in sig.provenance.notes)


@pytest.mark.parametrize(("chans", "msg"), [
    ([("I", 500, "uV", np.zeros(5000)), ("II", 250, "uV", np.zeros(2500))], "different sampling rates"),
    ([("I", 500, "uV", np.zeros(5000)), ("II", 500, "counts", np.zeros(5000))], "unknown units"),
])
def test_edf_mixed_rates_or_unknown_units_are_refused(tmp_path, chans, msg):
    _edf(tmp_path / "b.edf", chans)
    with pytest.raises(ValueError, match=msg):
        load_edf(tmp_path / "b.edf")


# --- CSV ---------------------------------------------------------------------------

def test_csv_bom_volts_and_one_column(tmp_path, ecg):
    f = tmp_path / "a.csv"
    f.write_text("﻿I,II\n" + "\n".join(f"{a / 1e6},{b / 1e6}" for a, b in ecg.signal[:2].T), encoding="utf-8")
    sig = load_csv(f, sampling_rate_hz=500, unit="V")
    assert sig.lead_names == ["I", "II"] and np.abs(sig.signal - ecg.signal[:2]).max() < 1
    g = tmp_path / "b.csv"
    g.write_text("II\n" + "\n".join(str(v) for v in ecg.signal[1]))
    assert load_csv(g, sampling_rate_hz=500).signal.shape == (1, ecg.n_samples)
    with pytest.raises(ValueError, match="unit"):
        load_csv(f, sampling_rate_hz=500, unit="mmHg")
    with pytest.raises(ValueError, match="positive"):
        load_csv(f, sampling_rate_hz=0)


# --- DICOM -------------------------------------------------------------------------

def test_dicom_padding_is_not_data(tmp_path, ecg):
    pytest.importorskip("pydicom")
    from cardiolens.layer1_ingestion import load_dicom
    from dicom_builder import group, write

    g = group(ecg.signal, ecg.lead_names, 500.0)
    counts = np.frombuffer(g.WaveformData, "<i2").reshape(-1, 12).copy()
    counts[-500:] = -32768                                        # the last second padded
    g.WaveformData, g.WaveformPaddingValue = counts.tobytes(), np.int16(-32768).tobytes()
    write(tmp_path / "p.dcm", [g])
    sig = load_dicom(tmp_path / "p.dcm")
    assert np.isnan(sig.signal[:, -500:]).all() and np.isfinite(sig.signal[:, :-500]).all()
    res = run_pipeline(sig, mode="A")
    assert res.measurements.heart_rate_bpm == pytest.approx(70, abs=3)
    assert "lvh" not in [f.code for f in res.findings.findings]


@pytest.mark.parametrize("broken", ["channels", "rate", "sensitivity", "junk"])
def test_dicom_inconsistent_header_is_refused(tmp_path, ecg, broken):
    pytest.importorskip("pydicom")
    from cardiolens.layer1_ingestion import load_dicom
    from dicom_builder import group, write

    g = group(ecg.signal, ecg.lead_names, 500.0)
    if broken == "channels":
        g.NumberOfWaveformChannels = 13                           # scrambled the leads silently
    elif broken == "rate":
        g.SamplingFrequency = 0
    elif broken == "sensitivity":
        g.ChannelDefinitionSequence[0].ChannelSensitivity = 0
    f = tmp_path / "x.dcm"
    write(f, [g])
    if broken == "junk":
        f.write_bytes(b"not a dicom file" * 20)
    with pytest.raises(ValueError):
        load_dicom(f)


# --- SCP-ECG -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def scp_units(ecg):
    return np.round(ecg.signal / 5.0).astype(np.int64)


@pytest.mark.parametrize(("kw", "msg"), [
    ({"ranges": [(10, 5)] * 12}, "invalid sample range"),
    ({"ranges": [(1, 5000)] * 6 + [(2501, 7500)] * 6}, "different samples"),
    ({"ranges": [(1, 4_000_000_000)] * 12}, r"declares \d+ samples"),     # was a 32-GiB allocation
    ({"sizes": [60000] * 12}, "past the end"),
])
def test_scp_lead_bounds_are_checked(tmp_path, ecg, scp_units, kw, msg):
    from test_scp_input import _record

    f = tmp_path / "x.scp"
    f.write_bytes(_record(scp_units, ecg.lead_names, **kw))
    with pytest.raises(ValueError, match=msg):
        load_scp(f)


def test_malformed_upload_is_a_400_not_a_500():
    from fastapi.testclient import TestClient

    from cardiolens.api.app import app

    client = TestClient(app, headers={"X-API-Key": "test-key"}, raise_server_exceptions=False)
    for name, data in (("ecg.edf", b"0" * 300), ("ecg.dcm", b"junk" * 50), ("ecg.csv", b"I,II\n1,2\n")):
        r = client.post("/api/analyze", data={"sampling_rate_hz": "500"}, files=[("files", (name, data, "x"))])
        assert r.status_code == 400, (name, r.status_code, r.text[:200])
