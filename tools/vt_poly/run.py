"""D277: polymorphic VT / torsade measurement — VFDB + CUDB 10-s windows through the site pipeline.

    python tools/vt_poly/run.py windows --out rows.jsonl [--shard k/n]
    python tools/vt_poly/run.py fresh --out rows.jsonl [--shard k/n]     # PTB-XL fresh-2 false alarms
    python tools/vt_poly/run.py score <windows.jsonl> [<fresh.jsonl>]

A window lies entirely in one annotated rhythm: VT, VFL, VF (VFDB "(VF"/"(VFIB", CUDB "[ ... ]"), ORG (any other
organised rhythm), NOISE, ASYS; up to 30 consecutive windows per record and class. No public base labels
polymorphic VT or torsade on its own, so the critical alarm is checked across VT / VFL / VF. Rows hold numbers
and codes only.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
WIN_S = 10.0
CAP = 30
FAMILY = {"ventricular_tachycardia", "wide_complex_tachycardia", "wide_qrs_tachycardia", "ventricular_fibrillation"}


def cls_of(r: str) -> str:
    if r == "VT":
        return "VT"
    if r == "VFL":
        return "VFL"
    if r in ("VF", "VFIB"):
        return "VF"
    if r in ("NOISE", "ASYS"):
        return r
    return "ORG"


def windows_of(db: str, path: str):
    import wfdb
    from vf_bench.run import _segments

    ann = wfdb.rdann(path, "atr")
    hdr = wfdb.rdheader(path)
    w = int(WIN_S * hdr.fs)
    per = Counter()
    for a, b, rh in _segments(db, ann, hdr.sig_len):
        c = cls_of(rh)
        for s in range(a, b - w + 1, w):
            if per[c] >= CAP:
                break
            per[c] += 1
            yield c, rh, s, w


def row_from(res) -> dict:
    m = res.measurements
    return {"codes": sorted(res.findings.codes()), "alerts": sorted({x.code for x in res.safety.alerts}),
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}


def run_windows(a) -> None:
    import numpy as np
    from cardiolens.pipeline import run_pipeline
    from vt_bench.run import load

    k, n = (int(v) for v in a.shard.split("/"))
    paths = sorted((db, str(p)[:-4]) for db in ("vfdb", "cudb") for p in (ROOT / "data/vf_bench" / db).glob("*.hea"))
    with a.out.open("w") as fh:
        for db, path in paths[k::n]:
            for c, rh, s0, w in windows_of(db, path):
                row = {"db": db, "rec": Path(path).name, "s0": s0, "cls": c, "rhythm": rh}
                try:
                    sig = load(path, s0, w)
                    if not np.isfinite(sig.signal).all():
                        continue
                    row.update(row_from(run_pipeline(sig, mode="A")))
                except Exception as ex:
                    row["error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
                fh.write(json.dumps(row) + "\n")
                fh.flush()


def run_fresh(a) -> None:
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline

    k, n = (int(v) for v in a.shard.split("/"))
    recs = json.loads((ROOT / "data/ptbxl_fresh2/manifest.json").read_text())["records"]
    with a.out.open("w") as fh:
        for r in recs[k::n]:
            row = {"id": r["ecg_id"], "labels": r["labels"]}
            try:
                sig = load_wfdb(str(ROOT / "data/ptbxl_fresh2" / r["filename_hr"]))
                row.update(row_from(run_pipeline(sig, mode="A")))
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
            fh.write(json.dumps(row) + "\n")


def score(files: list[Path]) -> None:
    rows = [json.loads(x) for x in files[0].open()]
    by = defaultdict(list)
    for r in rows:
        by[(r["db"], r["cls"])].append(r)
    print("db cls n alarm family_alarm  codes_with_alarm | missed: what named")
    for (db, c), g in sorted(by.items()):
        ok = [r for r in g if "error" not in r]
        al = [r for r in ok if set(r["alerts"]) & FAMILY]
        anya = [r for r in ok if r["alerts"]]
        how = Counter(next(iter(sorted(set(r["alerts"]) & FAMILY))) for r in al)
        miss = Counter(",".join(x for x in r["codes"] if "rhythm" in x or "fib" in x or "tachy" in x or
                                "irregular" in x or "flutter" in x or "quality" in x or "noise" in x) or "-"
                       for r in ok if r not in al)
        print(f"{db} {c} {len(g)} any={len(anya)} fam={len(al)} ({len(al) / max(1, len(ok)):.2f}) {dict(how)}"
              f" | {dict(miss.most_common(5))}")
    if len(files) > 1:
        fr = [json.loads(x) for x in files[1].open()]
        fr = [r for r in fr if "error" not in r]
        for lab in ("NORM", "STACH", "AFIB", "CLBBB", "CRBBB", "PVC", None):
            g = [r for r in fr if lab is None or lab in r["labels"]]
            bad = [r["id"] for r in g if set(r["alerts"]) & FAMILY]
            print(f"fresh2 {lab or 'all'}: {len(bad)}/{len(g)} family alarm {bad[:6]}")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["windows", "fresh", "score"])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.cmd == "windows":
        run_windows(a)
    elif a.cmd == "fresh":
        run_fresh(a)
    else:
        score(a.files)


if __name__ == "__main__":
    main()
