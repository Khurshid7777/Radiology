"""SD-166: which read lead makes a sheet's QRS (or QT) wrong. For a stand sheet whose interval error is large, each read lead
is replaced in turn by its true samples (aligned by the comparison's lag) and the intervals are measured again; the lead
after whose replacement the error falls below the threshold is the culprit. Stand cases only (development data).

    .venv/bin/python tools/scanner/qrs_culprit.py <cases dir> <bench cases.jsonl> [--slices a b ...] [--key qrs_ms]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def shifted_truth(truth: np.ndarray, t_start: float, lag_s: float, fs: float) -> tuple[np.ndarray, float]:
    """The true samples placed on the read's time axis: a read sample at time t matches the truth at t + lag."""
    start = t_start - lag_s
    if start < 0:
        cut = int(round(-start * fs))
        return truth[cut:], 0.0
    return truth, start


def culprits(errors: dict[str, float | None], base: float | None, threshold: float) -> list[str]:
    """Leads whose replacement brings an error above ``threshold`` below it (``errors``: lead -> error after replacing)."""
    if base is None or base <= threshold:
        return []
    return [k for k, v in errors.items() if v is not None and v <= threshold]


def where_worst(truth: np.ndarray, read: np.ndarray, r_peaks: np.ndarray, fs: float) -> float | None:
    """Seconds from the nearest R peak to the read's largest difference from the truth (both on one axis)."""
    n = min(len(truth), len(read))
    d = np.abs(truth[:n] - read[:n])
    if n == 0 or not np.isfinite(d).any() or len(r_peaks) == 0:
        return None
    i = int(np.nanargmax(d))
    return float((i - r_peaks[np.argmin(np.abs(r_peaks - i))]) / fs)


def main() -> None:
    from ecgscan.bench.cases import bench_config, load_case
    from ecgscan.bench.metrics import (
        _pick_output,
        clinical_measurements,
        compare_trace,
        truth_scan_result,
    )
    from ecgscan.contract.models import Trace
    from ecgscan.pipeline import scan

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", type=Path)
    ap.add_argument("bench", type=Path)
    ap.add_argument("--slices", nargs="*", default=None)
    ap.add_argument("--key", default="qrs_ms")
    ap.add_argument("--threshold", type=float, default=20.0)
    a = ap.parse_args()
    mcfg = bench_config()["metrics"]
    rows = [json.loads(line) for line in a.bench.read_text().splitlines() if line.strip()]
    todo = [r for r in rows if (not a.slices or r["slice"] in a.slices) and r["status"] != "rejected"
            and (r["interval_err_ms"].get(a.key) or 0) > a.threshold]
    found, kinds, unexplained = Counter(), Counter(), 0
    for row in todo:
        case = load_case(a.cases / row["slice"] / row["case_id"])
        res = scan(str(case.image_path))
        ref = clinical_measurements(truth_scan_result(case))[a.key]
        base = clinical_measurements(res)[a.key]
        base_err = None if base is None or ref is None else abs(base - ref)
        errs = {}
        fs = case.truth.fs
        for tr in case.truth.traces:
            out = _pick_output(res, tr.lead, tr.kind)
            if out is None:
                continue
            c = compare_trace(tr.samples_mv, tr.start_s, out.array(), out.start_s, fs, max_lag_s=mcfg["max_lag_s"],
                              min_overlap=mcfg["min_overlap"], snr_cap_db=mcfg["snr_cap_db"])
            if c is None:
                continue
            samples, start = shifted_truth(np.asarray(tr.samples_mv, float), tr.start_s, c["lag_s"], fs)
            alt = res.model_copy(deep=True)
            new = Trace.from_array(samples, start_s=start, fs=fs, lead=tr.lead, quality=1.0)
            if tr.kind == "rhythm":
                alt.rhythm = [new if t.lead == tr.lead else t for t in alt.rhythm]
            else:
                alt.leads = {**alt.leads, tr.lead: new}
            got = clinical_measurements(alt)[a.key]
            errs[f"{tr.lead}:{tr.kind}"] = None if got is None or ref is None else abs(got - ref)
        who = culprits(errs, base_err, a.threshold)
        print(json.dumps({"case": row["case_id"], "base_err": base_err, "culprits": who,
                          "best": min((v, k) for k, v in errs.items() if v is not None) if errs else None}), flush=True)
        if not who:
            unexplained += 1
        for k in who:
            found[k] += 1
            kinds[k.split(":")[1]] += 1
    print(json.dumps({"sheets": len(todo), "unexplained": unexplained, "culprit_leads": found.most_common(),
                      "kinds": dict(kinds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
