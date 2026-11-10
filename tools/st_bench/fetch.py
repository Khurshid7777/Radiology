"""D224: an ST bench from PTB-XL 1.0.3 — posterior / inferoposterior infarcts and the patterns they are
confused with, plus normal controls.

    python tools/st_bench/fetch.py --out data/ptbxl_st [--per-class 150] [--controls 400]

All PMI / IPMI / IPLMI records; a seeded sample of anteroseptal subendocardial injury and ischaemia (INJAS,
ISCAS), ST depression (STD_), incomplete / complete RBBB, RVH, WPW (dominant R in V1 mimics) and NORM controls.
Licensed data: the folder is git-ignored; only scripts and summaries go to git.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rhythm_bench"))
from fetch import BASE_URL, fetch_bytes  # tools/rhythm_bench/fetch.py

from cardiolens.validation.fetch_ptbxl import record_labels

FULL = ("PMI", "IPMI", "IPLMI")
SAMPLED = ("INJAS", "ISCAS", "STD_", "IRBBB", "CRBBB", "RVH", "WPW", "ASMI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=Path("data/ptbxl/ptbxl_database.csv"))
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--controls", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = list(csv.DictReader(a.db.open(encoding="utf-8")))
    labels = {r["ecg_id"]: record_labels(r) for r in rows}
    rng = random.Random(a.seed)
    chosen: dict[str, dict] = {}
    for code in FULL:
        for r in rows:
            if code in labels[r["ecg_id"]]:
                chosen[r["ecg_id"]] = r
    for code in SAMPLED:
        pool = [r for r in rows if code in labels[r["ecg_id"]] and r["ecg_id"] not in chosen]
        for r in rng.sample(pool, min(a.per_class, len(pool))):
            chosen[r["ecg_id"]] = r
    norm = [r for r in rows if {"NORM", "SR"} <= labels[r["ecg_id"]] and r["ecg_id"] not in chosen]
    for r in rng.sample(norm, min(a.controls, len(norm))):
        chosen[r["ecg_id"]] = r
    sel = list(chosen.values())
    print(f"[st] {len(sel)} records", file=sys.stderr, flush=True)

    def get(row: dict) -> None:
        dest = a.out / row["filename_hr"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            t = dest.with_suffix(ext)
            if not t.exists():
                t.write_bytes(fetch_bytes(BASE_URL + row["filename_hr"] + ext))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, sel))
    (a.out / "manifest.json").write_text(json.dumps({"source": BASE_URL, "seed": a.seed, "records": [
        {"ecg_id": int(r["ecg_id"]), "filename_hr": r["filename_hr"], "age": r.get("age"), "sex": r.get("sex"),
         "scp_codes": ast.literal_eval(r["scp_codes"]), "labels": sorted(record_labels(r))} for r in sel]},
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
