"""D268: run the pipeline over the SVT bench (tools/svt_bench/fetch.py) and score it.

    python tools/svt_bench/run.py --data data/svt_bench --out <dir> [--shard k/n]
    python tools/svt_bench/run.py score <dir> [<dir> ...]

A row per record: labels, codes, HR, QRS, alerts (no signal). Classes: SVT family (SVT, AVNRT, AVRT, AT, PSVT)
against sinus tachycardia (STACH), flutter (AFL) and sinus rhythm (SR).
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
SVT_CODES = {"supraventricular_tachycardia", "atrial_tachycardia"}
FAMILY = {"SVT", "AVNRT", "AVRT", "AT", "PSVT"}
SHOW = {"supraventricular_tachycardia", "atrial_tachycardia", "sinus_tachycardia", "atrial_flutter",
        "atrial_fibrillation", "possible_atrial_fibrillation", "wide_complex_tachycardia", "wide_qrs_tachycardia",
        "irregular_rhythm", "sinus_rhythm", "supraventricular_ectopy", "poor_quality_signal"}


def run(a) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline

    man = json.loads((a.data / "manifest.json").read_text())["records"]
    k, n = (int(v) for v in a.shard.split("/"))
    rows = a.out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    for rec in man[k::n]:
        dest = rows / f"{rec['name']}.json"
        if dest.exists():
            continue
        row = {"name": rec["name"], "labels": rec["labels"], "age": rec.get("age")}
        try:
            sig = load_wfdb(str(a.data / rec["folder"] / rec["name"]))
            sig.patient.age_years = rec.get("age")
            sig.patient.sex = {"Male": "male", "Female": "female"}.get(rec.get("sex") or "")
            res = run_pipeline(sig, mode="A")
            m = res.measurements
            row.update(codes=sorted(res.findings.codes()), hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms,
                       regular=m.rhythm_regular, alerts=[x.code for x in res.safety.alerts])
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
        dest.write_text(json.dumps(row))


def score(dirs: list[Path]) -> None:
    rows = [json.loads(p.read_text()) for d in dirs for p in (d / "rows").glob("*.json")]
    rows = [r for r in rows if "codes" in r]
    def member(r: dict, lab: str) -> bool:
        if lab == "SVT family":
            return bool(set(r["labels"]) & FAMILY)
        return lab in r["labels"] and (lab in FAMILY or not set(r["labels"]) & FAMILY)

    for name in ("SVT family", "SVT", "AVNRT", "AVRT", "AT", "PSVT", "STACH", "AFL", "SR"):
        rs = [r for r in rows if member(r, name)]
        if not rs:
            continue
        hit = sum(bool(set(r["codes"]) & SVT_CODES) for r in rs)
        c = Counter(x for r in rs for x in set(r["codes"]) & SHOW)
        hrs = sorted(r["hr"] for r in rs if r.get("hr"))
        print(f"{name:10s} n={len(rs):4d} SVT named {hit:4d} ({hit / len(rs):.2f}) HR median "
              f"{hrs[len(hrs) // 2] if hrs else None}  {dict(c.most_common(8))}")
    for lo, hi in ((0, 130), (130, 150), (150, 170), (170, 400)):
        rs = [r for r in rows if set(r["labels"]) & FAMILY and r.get("hr") and lo <= r["hr"] < hi]
        if rs:
            hit = sum(bool(set(r["codes"]) & SVT_CODES) for r in rs)
            print(f"  SVT family HR {lo}-{hi}: n={len(rs)} named {hit} ({hit / len(rs):.2f})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score([Path(p) for p in sys.argv[2:]])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
