"""D216: the atlas exam — ECG images from the owner's atlases through the site's photo route.

    python tools/atlas_exam/run.py [--cases 'data/books/exam/cases_*.csv'] [--out reports/atlas_exam/<name>.json]
                                   [--workers 4] [--only bank,yung]

Each image goes through ``digitize_photos`` + ``run_pipeline`` as an upload on the site (adult norms, no
age/sex). A case is scored on the author's main diagnoses mapped to finding codes (``our_codes``), with
the equivalences below (a family counts as found if any member is reported). The images and per-case
rows stay in data/books/exam/ (git-ignored); the summary JSON has counts only. This is a baseline exam:
no threshold is chosen on it (D216).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible"}

# A family is found if any member is reported (sinus rate names, "possible" forms, block grades).
FAMILIES = {
    "sinus_rhythm": {"sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia", "sinus_arrhythmia"},
    "sinus_bradycardia": {"sinus_bradycardia"},
    "sinus_tachycardia": {"sinus_tachycardia"},
    "atrial_fibrillation": {"atrial_fibrillation", "possible_atrial_fibrillation"},
    "atrial_flutter": {"atrial_flutter"},
    "st_elevation": {"st_elevation", "st_elevation_any"},
    "lvh": {"lvh", "lvh_probable"},
    "left_anterior_fascicular_block": {"left_anterior_fascicular_block"},
    "ventricular_ectopy": {"ventricular_ectopy", "ventricular_bigeminy", "ventricular_trigeminy"},
    "supraventricular_ectopy": {"supraventricular_ectopy", "atrial_bigeminy", "atrial_trigeminy"},
    "complete_av_block": {"complete_av_block", "slow_wide_escape_rhythm"},
    "wide_complex_tachycardia": {"wide_complex_tachycardia", "ventricular_tachycardia"},
}


def found(code: str, codes: set[str]) -> bool:
    return bool(FAMILIES.get(code, {code}) & codes)


def run_case(row: dict) -> dict:
    warnings.filterwarnings("ignore")
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    out = {k: row.get(k, "") for k in ("atlas", "case", "layout", "answer_ru", "our_codes", "adult")}
    png = ROOT / "data/books/exam/png" / Path(row["png"]).name if row.get("png") else None
    if png is None or not png.exists():
        return {**out, "status": "no_image"}
    try:
        sig = digitize_photos([png])
        res = run_pipeline(sig, mode="A")
    except Exception as e:  # a refusal is a result too
        return {**out, "status": "refused", "reason": str(e)[:160]}
    codes = res.findings.codes()
    m = res.measurements
    return {**out, "status": "gated" if codes & GATES else "read", "codes": sorted(codes),
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms,
            "route": "scanner" if (sig.provenance.notes or [""])[0].startswith("ecgscan") else "readers"}


def summarize(rows: list[dict]) -> dict:
    by_atlas: dict[str, dict] = {}
    per_code = defaultdict(lambda: [0, 0])
    for atlas in sorted({r["atlas"] for r in rows}):
        rs = [r for r in rows if r["atlas"] == atlas]
        read = [r for r in rs if r["status"] == "read"]
        exp = found_n = 0
        all_found = 0
        for r in read:
            want = [c for c in r["our_codes"].split(";") if c.strip()]
            got = set(r["codes"])
            hits = [found(c.strip(), got) for c in want]
            exp += len(want)
            found_n += sum(hits)
            all_found += bool(want) and all(hits)
            for c, h in zip(want, hits, strict=True):
                per_code[c.strip()][0] += h
                per_code[c.strip()][1] += 1
        by_atlas[atlas] = {"cases": len(rs), "read": len(read),
                           "refused": sum(r["status"] == "refused" for r in rs),
                           "gated": sum(r["status"] == "gated" for r in rs),
                           "no_image": sum(r["status"] == "no_image" for r in rs),
                           "diagnoses_found": f"{found_n}/{exp}",
                           "cases_all_found": f"{all_found}/{len(read)}"}
    n = len(rows)
    tot_exp = sum(int(v["diagnoses_found"].split("/")[1]) for v in by_atlas.values())
    tot_f = sum(int(v["diagnoses_found"].split("/")[0]) for v in by_atlas.values())
    return {"cases": n, "read": sum(r["status"] == "read" for r in rows),
            "read_rate": round(sum(r["status"] == "read" for r in rows) / max(1, n), 3),
            "diagnoses_found_of_read": f"{tot_f}/{tot_exp}",
            "recall_of_read": round(tot_f / max(1, tot_exp), 3),
            "by_atlas": by_atlas,
            "by_code": {c: f"{h}/{t}" for c, (h, t) in sorted(per_code.items(), key=lambda x: -x[1][1])}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "data/books/exam/cases_*.csv"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rows", type=Path, default=ROOT / "data/books/exam/results.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(a.cases)):
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh))
    if a.only:
        rows = [r for r in rows if r["atlas"] in a.only.split(",")]
    rows = [r for r in rows if r.get("layout") != "split"]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_case, rows))
    a.rows.parent.mkdir(parents=True, exist_ok=True)
    a.rows.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in res) + "\n", encoding="utf-8")
    s = summarize(res)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
