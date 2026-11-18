"""D304: site pipeline codes + the RR series (ms, numbers only) for Chapman AV blocks and PTB-XL AF / fresh sets.

    python tools/avb_bench/rr_dump.py {avb,ptb,fresh,fresh2} --out rows.jsonl [--shard k/n]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def items(which):
    if which == "avb":
        recs = json.loads((ROOT / "data/avb_bench/manifest.json").read_text())["records"]
        recs = [r for r in recs if set(r["labels"]) & {"2AVB", "2AVB1", "3AVB"}] + \
               [r for r in recs if set(r["labels"]) <= {"SB", "SR"}][::4]          # a quarter of the controls
        return [(str(ROOT / "data/avb_bench" / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age")) for r in recs]
    folder = {"ptb": "ptbxl_rhythm", "fresh": "ptbxl_fresh", "fresh2": "ptbxl_fresh2"}[which]
    recs = json.loads((ROOT / f"data/{folder}/manifest.json").read_text())["records"]
    keep = {"AFIB", "2AVB"} if which == "ptb" else {"AFIB"}
    return [(str(ROOT / f"data/{folder}" / r["filename_hr"]), r["ecg_id"], r["labels"], r.get("age"))
            for r in recs if set(r["labels"]) & keep]


def main():
    warnings.filterwarnings("ignore")
    import numpy as np

    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["avb", "ptb", "fresh", "fresh2"])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    with a.out.open("w") as fh:
        for path, ident, labels, age in items(a.which)[k::n]:
            row = {"id": ident, "labels": labels}
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                res = run_pipeline(sig, mode="A"); m = res.measurements
                rp = np.asarray(m.r_peaks_samples or [], float)
                row.update(codes=sorted(res.findings.codes()), alerts=sorted({x.code for x in res.safety.alerts}),
                           hr=m.heart_rate_bpm, rr=[round(float(x)) for x in np.diff(rp) * 1000 / sig.sampling_rate_hz])
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
