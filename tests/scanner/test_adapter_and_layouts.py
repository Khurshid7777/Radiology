"""ScanResult -> CardioLens SignalRepresentation; the layout registry."""

import numpy as np
import pytest

from ecgscan.contract import Trace
from ecgscan.contract.cardiolens_adapter import RejectedScanError, to_signal_representation
from ecgscan.layout import available_layouts, get_layout


def test_adapter_places_panels_and_rhythm_on_one_axis(result_factory):
    fs = 500.0
    leads = {"I": Trace.from_array(np.full(1250, 0.5), start_s=0.0, fs=fs, quality=1),
             "V1": Trace.from_array(np.full(1250, -1.0), start_s=5.0, fs=fs, quality=1),
             "II": Trace.from_array(np.full(1250, 9.0), start_s=0.0, fs=fs, quality=1)}
    rhythm = [Trace.from_array(np.full(5000, 0.2), start_s=0.0, fs=fs, lead="II", quality=1)]
    sig = to_signal_representation(result_factory(leads=leads, rhythm=rhythm))
    assert sig.n_samples == 5000 and sig.sampling_rate_hz == fs and sig.provenance.digitized
    assert sig.lead("I")[0] == pytest.approx(500.0) and sig.lead("I")[1300] == 0.0      # mV -> µV, zero outside
    assert sig.lead("V1")[2600] == pytest.approx(-1000.0)
    assert sig.provenance.lead_windows["V1"] == (5.0, 7.5)
    assert sig.lead("II")[4000] == pytest.approx(200.0)                                 # the strip wins
    assert sig.provenance.lead_windows["II"] == (0.0, 10.0)
    assert "aVR" not in sig.provenance.lead_windows and not sig.lead("aVR").any()
    assert any("digitized_from_image" in n for n in sig.provenance.notes)


def test_adapter_bridges_gaps_only_on_cardiolens_side_and_says_so(result_factory):
    x = np.linspace(0, 1, 11)
    x[3:6] = np.nan
    res = result_factory(leads={"I": Trace.from_array(x, start_s=0.0, fs=500, quality=1)})
    sig = to_signal_representation(res)
    assert np.isfinite(sig.lead("I")[:11]).all()
    assert any("bridged" in n and "I=3" in n for n in sig.provenance.notes)
    assert res.leads["I"].samples_mv[4] is None
    assert sig.provenance.sample_mask == {"I": [(0.006, 0.012)]}       # samples 3..5 at 500 Hz


def test_layer3_measures_no_beat_on_bridged_samples():
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer3_measure.measure import _lead_mask

    sig = generate_synthetic_ecg(heart_rate_bpm=60, seed=1)
    rpeaks = np.array([500, 1000, 1500, 2000])                          # 1, 2, 3, 4 s
    assert _lead_mask(sig, "V1", rpeaks, 500.0).all()
    sig.provenance.sample_mask = {"V1": [(2.1, 2.3)]}                  # a glare gap after the beat at 2.0 s
    assert _lead_mask(sig, "V1", rpeaks, 500.0).tolist() == [True, False, True, True]
    assert _lead_mask(sig, "V2", rpeaks, 500.0).all()


def test_adapter_refuses_rejected_scans(result_factory):
    with pytest.raises(RejectedScanError) as e:
        to_signal_representation(result_factory(status="rejected", reject_reasons=["GLARE"], leads={}))
    assert e.value.reasons == ["GLARE"]


def test_registry_has_v1_formats_with_complete_lead_sets():
    layouts = available_layouts()
    assert {"3x4+rhythm", "6x2", "12x1", "single"} <= set(layouts)
    twelve = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}
    for name in ("3x4+rhythm", "6x2", "12x1"):
        assert set(layouts[name].leads()) == twelve, name


def test_3x4_windows_follow_columns():
    lay = get_layout("3x4+rhythm")
    w = lay.lead_windows()
    assert w["I"] == (0.0, 2.5) and w["aVF"] == (2.5, 5.0) and w["V3"] == (5.0, 7.5) and w["V6"] == (7.5, 10.0)
    assert lay.record_s == 10.0 and lay.rhythm == ("II",) and lay.n_trace_rows == 4


def test_unknown_layout_is_an_error():
    with pytest.raises(KeyError):
        get_layout("5x5")
