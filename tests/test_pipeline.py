"""End-to-end pipeline + outputs smoke test (TZ §18 п.4, §11)."""

import json
import os

from cardiolens.config import load_rule_base
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.outputs import result_to_dict, write_all_outputs
from cardiolens.pipeline import run_pipeline


def test_rule_base_loads_and_builds():
    rb = load_rule_base()
    assert len(rb.rules) >= 4
    ids = {r.id for r in rb.rules}
    assert {"hyperkalemia_severity_chain", "electrode_reversal_la_ra"} <= ids


def test_end_to_end_normal():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    result = run_pipeline(sig, mode="A")
    d = result_to_dict(result)
    assert d["mode"] == "A"
    assert d["measurements"]["heart_rate_bpm"] > 0
    assert d["versions"]["rules_base"]
    assert isinstance(d["audit_log"], list) and d["audit_log"]
    # JSON is serialisable (TZ §11.1).
    assert json.loads(json.dumps(d, ensure_ascii=False))


def test_all_outputs_written(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6)
    result = run_pipeline(sig, mode="B")
    paths = write_all_outputs(result, str(tmp_path), stem="t")
    for key in ("json", "waveform_svg", "causal_graph_png", "pdf", "conclusion_txt"):
        assert os.path.exists(paths[key]), key
        assert os.path.getsize(paths[key]) > 0, key


def test_audit_log_traceability():
    # Traceability: audit log records each layer (TZ §11, §12).
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    joined = " ".join(result.audit_log)
    for tag in ("[L2]", "[L3]", "[L4]", "[L5]", "[L6]"):
        assert tag in joined


def test_pipeline_leaves_input_signal_untouched():
    # Preprocessing used to band-pass the caller's signal in place: a second run on the
    # same object saw the filtered signal (pacing spikes erased, R-peaks shifted).
    import numpy as np

    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    raw = sig.signal.copy()
    first = run_pipeline(sig, mode="A")
    assert np.array_equal(sig.signal, raw)
    second = run_pipeline(sig, mode="A")
    assert second.measurements.r_peaks_samples == first.measurements.r_peaks_samples
    assert second.findings.codes() == first.findings.codes()
