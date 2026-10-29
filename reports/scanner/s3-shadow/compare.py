"""S3, TZ 18.7: the illumination step (``flat_field``, SD-062) on every photo vs only on dark paper.

    python -m ecgscan bench --suite ablation_shadow --out <off> --cases <cases> --digitizers ecgscan
    ECGSCAN_CONFIG_DIR=<copy with quality.flatfield_below: 256> \\
        python -m ecgscan bench --suite ablation_shadow --out <on> --cases <cases> --digitizers ecgscan
    python reports/scanner/s3-shadow/compare.py <off>/cases.jsonl <on>/cases.jsonl > summary.json

The TZ's rule: on by default only if the median interval error improves and amplitudes do not
get worse. Per slice and overall: median interval error (RR, QRS, QT, PR), median amplitude error
(R), median r, cases read, cases worse / better by > 5 ms of interval error.
"""

import json
import sys

import numpy as np

INTERVALS = ("rr_ms", "qrs_ms", "qt_ms", "pr_ms")


def load(path):
    return {d["case_id"]: d for d in map(json.loads, open(path, encoding="utf-8"))}


def interval_err(d):
    e = d.get("interval_err_ms") or {}
    vals = [e[k] for k in INTERVALS if e.get(k) is not None]
    return float(np.mean(vals)) if vals else None


def stats(rows):
    read = [r for r in rows if r["status"] != "rejected" and r.get("r_median") is not None]
    ie = [interval_err(r) for r in read if interval_err(r) is not None]
    amp = [r["r_amp_err_median"] for r in read if r.get("r_amp_err_median") is not None]
    return dict(n=len(rows), read=len(read),
                interval_err_ms_median=round(float(np.median(ie)), 2) if ie else None,
                r_amp_err_mv_median=round(float(np.median(amp)), 4) if amp else None,
                r_median=round(float(np.median([r["r_median"] for r in read])), 4) if read else None,
                bad=sum(bool(r.get("bad")) for r in rows))


def main(off_path, on_path):
    off, on = load(off_path), load(on_path)
    ids = sorted(set(off) & set(on))
    out = {"cases": len(ids), "slices": {}}
    for sl in sorted({off[i]["slice"] for i in ids}) + ["all"]:
        sel = [i for i in ids if sl == "all" or off[i]["slice"] == sl]
        worse = better = 0
        for i in sel:
            a, b = interval_err(off[i]), interval_err(on[i])
            if a is not None and b is not None:
                worse += b > a + 5
                better += b < a - 5
        out["slices"][sl] = {"off": stats([off[i] for i in sel]), "on": stats([on[i] for i in sel]),
                             "on_worse_by_5ms": worse, "on_better_by_5ms": better}
    a, b = out["slices"]["all"]["off"], out["slices"]["all"]["on"]
    out["rule_met"] = bool(b["interval_err_ms_median"] < a["interval_err_ms_median"]
                           and b["r_amp_err_mv_median"] <= a["r_amp_err_mv_median"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
