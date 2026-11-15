"""D281: VF-detector metrics (per lead) for the tuning and acceptance windows — no pipeline, raw signal only.

    python tools/vt_poly/vfmetrics.py vtpoly <vt_poly rows.jsonl> --out m.jsonl [--shard k/n]
    python tools/vt_poly/vfmetrics.py mitdb --out m.jsonl [--shard k/n]        # 10-s windows, rhythm + V beats
    python tools/vt_poly/vfmetrics.py ptbxl <data dir> --out m.jsonl [--shard k/n]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def mts_of(sig):
    from cardiolens.arrhythmia.vf import detect_vf
    vf, mts = detect_vf(sig.signal.astype("float64"), sig.sampling_rate_hz)
    return vf, [None if m is None else {k: round(v, 4) for k, v in m.items()} for m in mts]


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vtpoly", "mitdb", "ptbxl"])
    ap.add_argument("src", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    from vt_bench.run import load
    with a.out.open("w") as fh:
        if a.cmd == "vtpoly":
            rows = [json.loads(x) for x in a.src.open()][k::n]
            for r in rows:
                path = str(ROOT / "data/vf_bench" / r["db"] / r["rec"])
                import wfdb
                fs = wfdb.rdheader(path).fs
                vf, mts = mts_of(load(path, r["s0"], int(10 * fs)))
                fh.write(json.dumps({**{x: r[x] for x in ("db", "rec", "s0", "cls")}, "vf": vf, "mts": mts}) + "\n")
        elif a.cmd == "mitdb":
            import wfdb
            from vt_bench.run import rhythm_spans
            paths = sorted(str(p)[:-4] for p in (ROOT / "data/mitdb").glob("*.hea"))[k::n]
            for path in paths:
                ann = wfdb.rdann(path, "atr")
                hdr = wfdb.rdheader(path)
                w = int(10 * hdr.fs)
                per = Counter()
                for s0e, s1e, rh in rhythm_spans(ann, hdr.sig_len):
                    for s0 in range(s0e, s1e - w + 1, w):
                        if per[rh] >= 20:
                            break
                        per[rh] += 1
                        sym = [sy for s, sy in zip(ann.sample, ann.symbol) if s0 <= s < s0 + w]
                        vf, mts = mts_of(load(path, s0, w))
                        fh.write(json.dumps({"db": "mitdb", "rec": Path(path).name, "s0": s0, "rhythm": rh,
                                             "nV": sum(x == "V" for x in sym), "vf": vf, "mts": mts}) + "\n")
        else:
            from cardiolens.layer1_ingestion import load_wfdb
            recs = json.loads((a.src / "manifest.json").read_text())["records"][k::n]
            for r in recs:
                vf, mts = mts_of(load_wfdb(str(a.src / r["filename_hr"])))
                fh.write(json.dumps({"id": r["ecg_id"], "labels": r["labels"], "vf": vf, "mts": mts}) + "\n")


if __name__ == "__main__":
    main()
