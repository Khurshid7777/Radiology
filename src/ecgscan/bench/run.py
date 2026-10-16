"""Run digitisers over a suite and aggregate by slice (TZ §11: one command, report in reports/)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ecgscan import PIPELINE_VERSION
from ecgscan.bench.cases import Case, bench_config, generate_suite, load_suite
from ecgscan.bench.digitizers import default_digitizers
from ecgscan.bench.metrics import (
    INTERVALS,
    CaseMetrics,
    TraceMetrics,
    clinical_measurements,
    score_case,
    truth_scan_result,
)


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def _median(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 3) if vals else None


def _rate(flags) -> float | None:
    flags = [bool(f) for f in flags if f is not None]
    return round(sum(flags) / len(flags), 3) if flags else None


def aggregate(rows: list[CaseMetrics]) -> dict:
    produced = [r for r in rows if r.status in ("ok", "needs_confirmation")]
    out = {
        "n": len(rows),
        "ok": sum(r.status == "ok" for r in rows),
        "needs_confirmation": sum(r.status == "needs_confirmation" for r in rows),
        "rejected": sum(r.status == "rejected" for r in rows),
        "errors": sum(r.status == "error" for r in rows),
        "r_median": _median([r.r_median for r in produced]),
        "rmse_mv_median": _median([r.rmse_median for r in produced]),
        "snr_db_median": _median([r.snr_median for r in produced]),
        "r_amp_err_mv_median": _median([r.r_amp_err_median for r in produced]),
        "leads_missing_mean": _mean([r.leads_missing for r in produced]),
        "speed_ok": _rate([r.speed_ok for r in rows if r.status != "error"]),
        "gain_ok": _rate([r.gain_ok for r in rows if r.status != "error"]),
        "px_per_mm_err_median": _median([r.px_per_mm_rel_err for r in rows]),
        "bad": _rate([r.bad for r in rows]),
        # A bad case that came back "ok" is the dishonest outcome (TZ §11): errors and
        # explicit refusals/confirmation requests are honest.
        "unflagged_bad": _rate([r.bad and r.status == "ok" for r in rows]),
        "time_s_median": _median([r.elapsed_s for r in rows]),
        "time_s_max": round(max((r.elapsed_s or 0.0) for r in rows), 3) if rows else None,
    }
    for k in INTERVALS:
        out[f"{k}_mae"] = _mean([r.interval_err_ms.get(k) for r in produced])
        measured_ref = [r for r in produced if k in r.interval_err_ms]
        out[f"{k}_unmeasured"] = _rate([k in r.unmeasured for r in measured_ref]) if measured_ref else None
    return out


def acceptance(agg: dict, acc: dict) -> dict:
    """TZ §11 criteria on one slice aggregate: {criterion: (value, threshold, passed)}."""
    checks = {
        "RR MAE, мс": (agg["rr_ms_mae"], acc["interval_mae_ms"], "<="),
        "QRS MAE, мс": (agg["qrs_ms_mae"], acc["interval_mae_ms"], "<="),
        "QT MAE, мс": (agg["qt_ms_mae"], acc["interval_mae_ms"], "<="),
        "SNR, дБ (медиана)": (agg["snr_db_median"], acc["snr_db_min"], ">="),
        "r (медиана)": (agg["r_median"], acc["r_min"], ">="),
        "ошибка R, мВ (медиана)": (agg["r_amp_err_mv_median"], acc["r_amp_err_mv_median"], "<="),
        "время, с (макс.)": (agg["time_s_max"], acc["time_s_max"], "<="),
        "плохие без пометки": (agg["unflagged_bad"], acc["unflagged_bad_max"], "<="),
    }
    out = {}
    for name, (v, thr, op) in checks.items():
        passed = None if v is None else (v <= thr if op == "<=" else v >= thr)
        out[name] = {"value": v, "threshold": f"{op} {thr}", "passed": passed}
    return out


def _row_from_json(d: dict) -> CaseMetrics:
    d = {k: v for k, v in d.items() if k != "reference"}
    d["traces"] = [TraceMetrics(**t) for t in d["traces"]]
    return CaseMetrics(**d)


def _git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=False, timeout=10).stdout.strip() or None
    except OSError:
        return None


def run_bench(suite: str, out_dir: str | Path, *, cases_dir: str | Path | None = None, digitizers=None,
              cases_per_slice: int | None = None, slices: list[str] | None = None,
              cases: list[Case] | None = None, resume: bool = False, log=print) -> dict:
    """``cases`` (e.g. ``image_record_cases``) replaces the synthetic suite when given."""
    cfg = bench_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    skipped: dict[str, str] = {}
    if cases is not None:
        pass
    elif cases_dir and Path(cases_dir).exists() and any(Path(cases_dir).rglob("case.json")):
        cases = load_suite(cases_dir)
        if slices:
            cases = [c for c in cases if c.slice in slices]
    else:
        cases, skipped = generate_suite(suite, cases_dir or out_dir / "cases",
                                        cases_per_slice=cases_per_slice, slices=slices)
    log(f"{len(cases)} cases ({time.perf_counter() - t0:.0f} s to prepare); skipped slices: {skipped or '—'}")
    digitizers = digitizers if digitizers is not None else default_digitizers(cfg)
    active, unavailable = [], {}
    for d in digitizers:
        ok, why = d.available()
        (active.append(d) if ok else unavailable.__setitem__(d.name, why))
    rows: list[CaseMetrics] = []
    done: set[tuple[str, str]] = set()
    jsonl = out_dir / "cases.jsonl"
    if resume and jsonl.exists():
        # A long run killed midway (container restart) continues where it stopped.
        kept = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                cm = _row_from_json(json.loads(line))
            except (ValueError, TypeError, KeyError):
                continue                                   # a half-written last line is redone
            rows.append(cm)
            kept.append(line)
            done.add((cm.case_id, cm.digitizer))
        jsonl.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        log(f"resume: {len(done)} case x digitiser rows already scored")
    with jsonl.open("a" if resume else "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases):
            todo = [d for d in active if (case.case_id, d.name) not in done]
            if not todo:
                continue
            reference = clinical_measurements(truth_scan_result(case))
            hint = {"speed_mm_s": case.truth.speed_mm_s, "gain_mm_mv": case.truth.gain_mm_mv}
            for d in todo:
                run = d.digitize(case.image_path, truth_hint=hint)
                cm = score_case(case, d.name, run, reference, cfg)
                rows.append(cm)
                rec = asdict(cm)
                rec["reference"] = {k: (None if v is None else round(v, 1)) for k, v in reference.items()}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            log(f"[{i + 1}/{len(cases)}] {case.case_id}: "
                + "; ".join(f"{r.digitizer} {r.status} r={r.r_median} QRSΔ={r.interval_err_ms.get('qrs_ms')}"
                            for r in rows[-len(todo):]))
    by: dict[tuple[str, str], list[CaseMetrics]] = defaultdict(list)
    for r in rows:
        by[(r.digitizer, r.slice)].append(r)
    slices_seen = list(dict.fromkeys(c.slice for c in cases))
    summary = {
        "suite": suite, "pipeline_version": PIPELINE_VERSION, "git": _git_rev(), "python": platform.python_version(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_cases": len(cases),
        "digitizers": [d.name for d in active], "unavailable": unavailable, "skipped_slices": skipped,
        "slices": slices_seen,
        "by_slice": {d.name: {s: aggregate(by[(d.name, s)]) for s in slices_seen if by[(d.name, s)]} for d in active},
        "overall": {d.name: aggregate([r for r in rows if r.digitizer == d.name]) for d in active},
        "acceptance": {}, "wall_s": round(time.perf_counter() - t0, 1),
    }
    for d in active:
        acc_rows = [r for r in rows if r.digitizer == d.name and r.slice in cfg["acceptance"]["slices"]]
        if acc_rows:
            summary["acceptance"][d.name] = acceptance(aggregate(acc_rows), cfg["acceptance"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    from ecgscan.bench.report import write_report

    write_report(summary, out_dir / "report.md")
    return summary
