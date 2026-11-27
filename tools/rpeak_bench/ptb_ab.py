"""D320: every PTB-XL fresh record with and without the edge padding of the R detector (HR, codes, alerts only)."""
import argparse
import importlib
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
    M = importlib.import_module("cardiolens.layer3_measure.measure")
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sets", default="ptbxl_fresh,ptbxl_fresh2"); ap.add_argument("--parity", type=int, default=-1)
    ap.add_argument("--keep", type=float, default=0.0)
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    items = []
    for d in a.sets.split(","):
        for r in json.loads((ROOT / f"data/{d}/manifest.json").read_text())["records"]:
            if d == "avb_bench":       # Chapman controls (D322 acceptance on untouched data)
                if set(r["labels"]) <= {"SB", "SR"}:
                    items.append((d, dict(r, ecg_id=r["name"], filename_hr=r["folder"] + r["name"])))
            elif a.parity < 0 or int(r["ecg_id"]) % 2 == a.parity:
                items.append((d, r))
    with a.out.open("w") as fh:
        for d, r in items[k::n]:
            row = {"id": r["ecg_id"], "labels": r["labels"]}
            try:
                sig = load_wfdb(str(ROOT / f"data/{d}" / r["filename_hr"]))
                try:
                    sig.patient.age_years = float(r.get("age"))
                except (TypeError, ValueError):
                    pass
                for pad in (0.0, 1.0):
                    M.R_EDGE_PAD_S = pad
                    res = run_pipeline(sig, mode="A")
                    row[str(pad)] = {"hr": res.measurements.heart_rate_bpm, "codes": sorted(res.findings.codes()),
                                     "alerts": sorted({x.code for x in res.safety.alerts})}
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
