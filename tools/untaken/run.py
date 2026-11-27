"""D312: run the site pipeline over data/untaken_bench; rows hold codes, alerts, HR, QRS (numbers only)."""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    recs = json.loads((ROOT / "data/untaken_bench/manifest.json").read_text())["records"][k::n]
    with a.out.open("w") as fh:
        for r in recs:
            row = {"name": r["name"], "labels": r["labels"], "dx": r["dx"]}
            try:
                sig = load_wfdb(str(ROOT / "data/untaken_bench" / r["folder"] / r["name"]))
                sig.patient.age_years = r.get("age")
                res = run_pipeline(sig, mode="A"); m = res.measurements
                row.update(codes=sorted(res.findings.codes()), alerts=sorted({x.code for x in res.safety.alerts}),
                           hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms, pr=m.intervals.pr_ms, rr_cv=m.rr_cv)
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
