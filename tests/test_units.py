"""Amplitude units of loaded files (D119)."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline


def test_amplitude_a_thousand_times_too_large_withholds_everything():
    sig = generate_synthetic_ecg(heart_rate_bpm=80, st_elevation_uv=250, st_leads=["II", "III", "aVF"], seed=5)
    big = sig.model_copy(update={"signal": sig.signal * 1000.0})
    r = run_pipeline(big, mode="A")
    assert r.findings.codes() == {"amplitude_implausible"}
    assert not r.safety.alerts


def test_csv_in_mv_read_as_uv_hints_at_units(tmp_path):
    from cardiolens.layer1_ingestion.parsers import load_csv

    sig = generate_synthetic_ecg(heart_rate_bpm=70, seed=1)
    path = tmp_path / "ecg.csv"
    np.savetxt(path, (sig.signal / 1000.0).T, delimiter=",", header=",".join(sig.lead_names), comments="")
    r = run_pipeline(load_csv(path, sampling_rate_hz=sig.sampling_rate_hz), mode="A")    # unit defaults to µV
    f = next(f for f in r.findings.findings if f.code == "no_ecg_signal")
    assert "единицы" in f.label_ru


@pytest.mark.parametrize("dim,factor", [("uV", 1.0), ("mV", 1000.0)])
def test_edf_uses_the_declared_dimension(tmp_path, dim, factor):
    pyedflib = pytest.importorskip("pyedflib")
    from cardiolens.layer1_ingestion.parsers import load_edf

    x = np.sin(np.linspace(0, 20, 2500)) * (1500.0 if dim == "uV" else 1.5)
    path = str(tmp_path / "e.edf")
    lim = 5000.0 / factor
    with pyedflib.EdfWriter(path, 1) as w:
        w.setSignalHeader(0, {"label": "II", "dimension": dim, "sample_frequency": 250,
                              "physical_max": lim, "physical_min": -lim, "digital_max": 32767, "digital_min": -32768})
        w.writeSamples([x])
    s = load_edf(path)
    assert float(np.max(s.signal)) == pytest.approx(1500.0, rel=0.02)
