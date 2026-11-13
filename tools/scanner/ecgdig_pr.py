"""R7 (SD-110): the PR interval on the ecgdig pages — ours against the true signal's, with the two
PR estimates of CardioLens D250 (single-lead and multi-lead onset) kept for the stability flag.

    python tools/scanner/ecgdig_pr.py --data data/ecgdig --out <dir> [--shard k/n]
    python tools/scanner/ecgdig_pr.py --out <dir> --summary

One JSON per page in ``<dir>/rows`` (resumable). The set is test-only: nothing is tuned on it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "ecgdig_bench"))

HIDING = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible"}   # no number is shown under these (D178)


def _pr(m) -> dict:
    return {"pr": m.intervals.pr_ms, "pr_multi": m.pr_multilead_ms, "pr_beats": m.pr_beats,
            "pr_std": m.pr_std_ms, "p_ratio": m.p_onset_ratio,
            "pr_dwt": any(n.startswith("PR from dwt") for n in m.notes),
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}


def run(a) -> None:
    warnings.filterwarnings("ignore")
    from run import truth_signal  # tools/ecgdig_bench/run.py

    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    fs_of = {r["id"]: float(r["fs"]) for r in csv.DictReader((a.data / "train.csv").open())}
    recs = sorted(p.name for p in (a.data / "train").iterdir() if p.is_dir())
    k, n = (int(v) for v in a.shard.split("/"))
    rows = a.out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    for rec in recs[k::n]:
        truth, _ = truth_signal(a.data / "train" / rec / f"{rec}.csv", fs_of[rec])
        t = _pr(run_pipeline(truth, mode="A").measurements)
        for img in sorted((a.data / "train" / rec).glob(f"{rec}-v*.*")):
            cond = img.stem.split("-")[1]
            out = rows / f"{rec}-{cond}.json"
            if out.exists():
                continue
            row = {"record": rec, "cond": cond, **{f"{key}_true": v for key, v in t.items()}}
            try:
                sig = digitize_photos([img])
                res = run_pipeline(sig, mode="A")
                notes = sig.provenance.notes or []
                scan = next((x for x in notes if x.startswith("ecgscan")), "")
                row.update(read=True, **_pr(res.measurements),
                           gates=sorted(res.findings.codes() & HIDING), codes=sorted(res.findings.codes()),
                           route="scanner" if scan else "readers",
                           status=scan.split("status=")[1].split()[0] if "status=" in scan else None,
                           warnings=next((x[10:] for x in notes if x.startswith("warnings: ")), ""))
            except Exception as e:  # a refusal is a result too
                row.update(read=False, error=str(e)[:200])
            out.write_text(json.dumps(row, ensure_ascii=False))
            print(f"{rec} {cond} PR {row.get('pr')} / {row['pr_true']} multi {row.get('pr_multi')}",
                  file=sys.stderr, flush=True)


def summary(out: Path, tol: float = 20.0, unstable_ms: float | None = None) -> dict:
    """Per variant: pages read, with a true PR, ours within ``tol``; ours shown (no hiding gate, and — with
    ``unstable_ms`` — the two estimates agree) and off by more than ``tol``: the silent PR errors."""
    by = defaultdict(list)
    for f in sorted((out / "rows").glob("*.json")):
        r = json.loads(f.read_text())
        by[r["cond"]].append(r)
    table = {}
    for cond in sorted(by):
        rs = [r for r in by[cond] if r.get("read") and r.get("pr_true") is not None]

        def shown(r):
            if r.get("gates") or r.get("pr") is None:
                return False
            if unstable_ms is not None and r.get("pr_multi") is not None:
                return abs(r["pr_multi"] - r["pr"]) <= unstable_ms
            return True

        off = [r for r in rs if r.get("pr") is None or abs(r["pr"] - r["pr_true"]) > tol]
        table[cond] = {"n": len(by[cond]), "read_with_true_pr": len(rs), "pr_ok": len(rs) - len(off),
                       "pr_shown": sum(shown(r) for r in rs),
                       "pr_shown_wrong": sum(shown(r) and abs(r["pr"] - r["pr_true"]) > tol for r in rs),
                       "pr_unmeasured": sum(r.get("pr") is None for r in rs)}
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--unstable-ms", type=float, default=None)
    a = ap.parse_args()
    if a.summary:
        for cond, row in summary(a.out, unstable_ms=a.unstable_ms).items():
            print(cond, json.dumps(row))
        return
    run(a)


if __name__ == "__main__":
    main()
