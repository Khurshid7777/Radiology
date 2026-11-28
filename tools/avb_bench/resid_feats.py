"""D367: current-core codes + D366 P positions (model C, with and without the P-P grid) per record; numbers and codes only.

    python tools/avb_bench/resid_feats.py avb|af|ptb --model <c.onnx> --out <file.jsonl> [--shard k/n]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/p_detector"))
LEADS = ("II", "V1", "III", "aVF")


def main():
    warnings.filterwarnings("ignore")
    import onnxruntime as ort

    import cnn_resid as C
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    from cnn import prep_signal
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["avb", "af", "ptb"])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--parity", default="even", choices=["even", "odd"])   # af only: even = tuning (D367), odd = acceptance (D368)
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(a.model), o, providers=["CPUExecutionProvider"])
    if a.which in ("avb", "af"):
        base = ROOT / ("data/avb_bench" if a.which == "avb" else "data/af_bench")
        recs = json.loads((base / "manifest.json").read_text())["records"]
        if a.which == "af":
            want = 0 if a.parity == "even" else 1
            recs = [r for r in recs if int("".join(c for c in r["name"] if c.isdigit())) % 2 == want]
        items = [(str(base / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age")) for r in recs]
    else:
        recs = json.loads((ROOT / "data/ptbxl_rhythm/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/ptbxl_rhythm" / r["filename_hr"]), r["ecg_id"], r["labels"], r.get("age")) for r in recs]
    with a.out.open("w") as fh:
        for path, ident, labels, age in items[k::n]:
            row = {"id": ident, "labels": labels}
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                res = run_pipeline(sig, mode="A")
                m, s = res.measurements, res.signal
                fs = float(s.sampling_rate_hz)
                rp = np.asarray(m.r_peaks_samples or [], int)
                sc = C.FS / fs
                rp5 = (rp * sc).astype(int)
                row.update(codes=sorted(res.findings.codes()), alerts=sorted({x.code for x in res.safety.alerts}),
                           hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms, pr=m.intervals.pr_ms, rp=rp5.tolist())
                for ln in LEADS:
                    if ln not in s.lead_names:
                        continue
                    x = prep_signal(np.asarray(s.lead(ln), float), fs)
                    raw = C.peaks_from_prob(C.predict(sess, C.two(x, rp5)), x, 0.5)
                    row[ln] = {"p": raw.tolist(), "grid": C.grid_gate(raw, 0.7).tolist() if len(raw) >= 3 else raw.tolist()}
            except Exception as ex:
                row["error"] = str(ex)[:80]
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
