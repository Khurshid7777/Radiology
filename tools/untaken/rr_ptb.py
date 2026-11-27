"""D318: RR series and codes for every record of a PTB-XL stand (numbers only).

    python tools/untaken/rr_ptb.py {ptbxl_rhythm,ptbxl_fresh,ptbxl_fresh2} --out rows.jsonl [--shard k/n]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("which"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    recs = json.loads((ROOT / f"data/{a.which}/manifest.json").read_text())["records"][k::n]
    with a.out.open("w") as fh:
        for r in recs:
            row = {"id": r["ecg_id"], "labels": r["labels"]}
            try:
                sig = load_wfdb(str(ROOT / f"data/{a.which}" / r["filename_hr"]))
                try:
                    sig.patient.age_years = float(r.get("age"))
                except (TypeError, ValueError):
                    pass
                res = run_pipeline(sig, mode="A"); m = res.measurements
                rp = np.asarray(m.r_peaks_samples or [], float)
                row.update(codes=sorted(res.findings.codes()), hr=m.heart_rate_bpm,
                           rr=[round(float(x)) for x in np.diff(rp) * 1000 / res.signal.sampling_rate_hz])
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
