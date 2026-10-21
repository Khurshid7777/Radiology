"""Bench metrics and harness: perfect, wrong-speed and refusing digitisers are scored right."""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ecgscan.bench.cases import bench_config, generate_suite, load_suite
from ecgscan.bench.digitizers import CardiolensV0, DigitizerRun, ExternalCommand
from ecgscan.bench.metrics import (
    clinical_measurements,
    compare_trace,
    score_case,
    truth_scan_result,
)
from ecgscan.bench.run import aggregate, run_bench
from ecgscan.contract import Trace

pytest.importorskip("cv2")


def test_compare_trace_identity_shift_and_gaps():
    t = np.arange(1250) / 500.0
    x = np.sin(2 * np.pi * 1.3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
    same = compare_trace(x, 0.0, x, 0.0, 500, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=60)
    assert same["r"] == 1.0 and same["snr_db"] == 60 and same["lag_s"] == 0.0 and same["rmse_mv"] == 0.0
    shifted = compare_trace(x, 0.0, x[50:], 0.0, 500, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=60)
    assert shifted["lag_s"] == pytest.approx(0.1) and shifted["r"] > 0.999
    holes = x.copy()
    holes[100:400] = np.nan
    gap = compare_trace(x, 0.0, holes, 0.0, 500, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=60)
    assert gap["coverage"] == pytest.approx(0.76) and gap["r"] == 1.0          # gaps cost coverage, not fake data
    halved = compare_trace(x, 0.0, x * 0.5, 0.0, 500, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=60)
    assert halved["r"] == pytest.approx(1.0) and halved["snr_db"] == pytest.approx(6.02, abs=0.01)
    assert compare_trace(x, 0.0, x[:200], 0.0, 500, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=60) is None


@pytest.fixture(scope="module")
def smoke_cases(tmp_path_factory):
    out = tmp_path_factory.mktemp("cases")
    cases, skipped = generate_suite("smoke", out, cases_per_slice=1, slices=["clean_scan", "speed_50"])
    assert not skipped and len(cases) == 2
    return out, cases


def test_cases_roundtrip_and_reproducible(smoke_cases, tmp_path):
    out, cases = smoke_cases
    loaded = {c.case_id: c for c in load_suite(out)}
    for c in cases:
        lc = loaded[c.case_id]
        assert lc.meta == json.loads(json.dumps(c.meta))
        for a, b in zip(c.truth.traces, lc.truth.traces, strict=True):
            assert a.lead == b.lead and np.allclose(a.samples_mv, b.samples_mv, atol=1e-5)
    again, _ = generate_suite("smoke", tmp_path, cases_per_slice=1, slices=["clean_scan"])
    assert again[0].image_path.read_bytes() == cases[0].image_path.read_bytes()


def test_truth_scores_perfectly(smoke_cases):
    _, cases = smoke_cases
    case = cases[0]
    ref = clinical_measurements(truth_scan_result(case))
    assert ref["heart_rate_bpm"] and ref["qrs_ms"]
    cm = score_case(case, "perfect", DigitizerRun(truth_scan_result(case), None, 0.1), ref, bench_config())
    assert cm.status == "ok" and cm.r_median == 1.0 and cm.leads_missing == 0
    assert all(v == 0.0 for v in cm.interval_err_ms.values()) and cm.bad is False
    assert cm.speed_ok and cm.gain_ok


def test_wrong_speed_halves_the_rate_and_is_caught(smoke_cases):
    """TZ §5.4: 50 mm/s read as 25 mm/s gives half the heart rate — the bench must flag it."""
    _, cases = smoke_cases
    case = next(c for c in cases if c.slice == "speed_50")
    truth = truth_scan_result(case)
    ref = clinical_measurements(truth)
    # A digitiser assuming 25 mm/s on a 50 mm/s page: every trace is twice as long in time.
    def stretch(tr: Trace) -> Trace:
        x = tr.array()
        y = np.interp(np.arange(2 * len(x)) / 2.0, np.arange(len(x)), x)
        return Trace.from_array(y, start_s=2 * tr.start_s, fs=500, lead=tr.lead, quality=1.0)

    wrong = truth.model_copy(update={
        "leads": {k: stretch(v) for k, v in truth.leads.items()}, "rhythm": [stretch(t) for t in truth.rhythm],
        "calibration": truth.calibration.model_copy(update={"paper_speed_mm_s": 25, "speed_source": None})})
    got = clinical_measurements(wrong)
    assert got["heart_rate_bpm"] == pytest.approx(ref["heart_rate_bpm"] / 2, rel=0.05)
    cm = score_case(case, "assumes-25", DigitizerRun(wrong, None, 0.1), ref, bench_config())
    assert cm.speed_ok is False and cm.bad is True
    assert aggregate([cm])["unflagged_bad"] == 1.0        # returned "ok": the dishonest outcome is counted


def test_refusals_and_errors_are_honest(smoke_cases):
    _, cases = smoke_cases
    case = cases[0]
    ref = clinical_measurements(truth_scan_result(case))
    rejected = truth_scan_result(case).model_copy(update={"status": "rejected", "reject_reasons": ["BLUR"]})
    rows = [score_case(case, "x", DigitizerRun(rejected, None, 0.1), ref, bench_config()),
            score_case(case, "x", DigitizerRun(None, "ValueError: no grid", 0.1), ref, bench_config())]
    agg = aggregate(rows)
    assert agg["rejected"] == 1 and agg["errors"] == 1 and agg["unflagged_bad"] == 0.0


def test_legacy_digitizer_adapter_on_a_clean_page(smoke_cases):
    _, cases = smoke_cases
    case = cases[0]
    run = CardiolensV0().digitize(case.image_path)
    assert run.error is None and run.result.status == "ok"
    assert run.result.source.pipeline_version == "cardiolens-v0"
    # The legacy page digitiser reports lead II only as the 10-s rhythm strip.
    assert len(run.result.leads) == 11 and [t.lead for t in run.result.rhythm] == ["II"]
    ref = clinical_measurements(truth_scan_result(case))
    cm = score_case(case, "cardiolens-v0", run, ref, bench_config())
    assert cm.r_median > 0.85 and cm.gain_ok


def test_external_digitizer_contract(smoke_cases, tmp_path, monkeypatch):
    _, cases = smoke_cases
    case = cases[0]
    ext = ExternalCommand("fake", {"enabled": False, "env": "FAKE_DIGITIZER_CMD", "licence": "MIT"})
    assert ext.available() == (False, "disabled in bench.yaml (licence: MIT)")
    ext.cfg["enabled"] = True
    assert ext.available()[0] is False                     # env var not set
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(truth_scan_result(case).to_json_dict()), encoding="utf-8")
    script = tmp_path / "fake.py"
    script.write_text(textwrap.dedent(f"""
        import shutil, sys
        shutil.copy({str(payload)!r}, sys.argv[2] + "/result.json")
    """), encoding="utf-8")
    monkeypatch.setenv("FAKE_DIGITIZER_CMD", f"{sys.executable} {script} {{image}} {{out_dir}}")
    assert ext.available() == (True, "")
    run = ext.digitize(case.image_path)
    assert run.error is None and run.result.layout.name == "3x4+rhythm"
    monkeypatch.setenv("FAKE_DIGITIZER_CMD", f"{sys.executable} -c 'import sys; sys.exit(3)' {{image}}")
    assert ext.digitize(case.image_path).error.startswith("exit 3")


def test_run_bench_writes_report(smoke_cases, tmp_path):
    out, _ = smoke_cases

    class Perfect:
        name = "perfect"

        def available(self):
            return True, ""

        def digitize(self, image, *, truth_hint=None):
            case = next(c for c in load_suite(out) if c.image_path == Path(image))
            return DigitizerRun(truth_scan_result(case), None, 0.01)

    summary = run_bench("smoke", tmp_path / "rep", cases_dir=out, digitizers=[Perfect()], log=lambda *_: None)
    assert summary["n_cases"] == 2 and summary["digitizers"] == ["perfect"]
    agg = summary["overall"]["perfect"]
    assert agg["r_median"] == 1.0 and agg["qrs_ms_mae"] == 0.0 and agg["unflagged_bad"] == 0.0
    report = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
    assert "Бенчмарк сканера ЭКГ" in report and "`perfect`" in report
    rows = (tmp_path / "rep" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and json.loads(rows[0])["reference"]["heart_rate_bpm"]


def test_image_folder_pairs_kit_and_database_names(synth_mv, tmp_path):
    """ECG-Image-Kit (<rec>-0.png) and ECG-Image-Database (<rec>-0001.png) next to WFDB records."""
    import wfdb

    from ecgscan.bench.cases import image_record_cases

    leads = list(synth_mv)
    data = np.stack([synth_mv[k] for k in leads], axis=1)
    for rec in ("00001_hr", "00002_hr"):
        wfdb.wrsamp(rec, fs=500, units=["mV"] * 12, sig_name=leads, p_signal=data, write_dir=str(tmp_path),
                    fmt=["16"] * 12)
    (tmp_path / "00001_hr-0.png").write_bytes(b"x")
    (tmp_path / "00002_hr-0001.png").write_bytes(b"x")
    (tmp_path / "orphan-0.png").write_bytes(b"x")
    cases = image_record_cases(tmp_path, speed_mm_s=50.0)
    assert [c.case_id for c in cases] == ["00001_hr-0", "00002_hr-0001"]
    c = cases[0]
    assert c.truth.speed_mm_s == 50.0 and c.meta["assumed"]["layout"] == "3x4+rhythm"
    v4 = next(t for t in c.truth.traces if t.lead == "V4")
    assert v4.start_s == 7.5 and np.allclose(v4.samples_mv, synth_mv["V4"][3750:5000], atol=2e-3)
    assert next(t for t in c.truth.traces if t.kind == "rhythm").samples_mv.size == 5000


def test_resume_scores_only_missing_rows(smoke_cases, tmp_path):
    out, _ = smoke_cases
    calls = []

    class Counting:
        name = "counting"

        def available(self):
            return True, ""

        def digitize(self, image, *, truth_hint=None):
            calls.append(image)
            case = next(c for c in load_suite(out) if c.image_path == Path(image))
            return DigitizerRun(truth_scan_result(case), None, 0.01)

    rep = tmp_path / "rep"
    run_bench("smoke", rep, cases_dir=out, digitizers=[Counting()], log=lambda *_: None)
    lines = (rep / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    (rep / "cases.jsonl").write_text(lines[0] + "\n" + lines[1][:40], encoding="utf-8")   # killed mid-write
    calls.clear()
    summary = run_bench("smoke", rep, cases_dir=out, digitizers=[Counting()], resume=True, log=lambda *_: None)
    assert len(calls) == 1 and summary["overall"]["counting"]["n"] == 2
    rows = (rep / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and all(json.loads(r)["digitizer"] == "counting" for r in rows)
