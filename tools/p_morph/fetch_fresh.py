"""D271: fresh PTB-XL records for acceptance — never in data/ptbxl or data/ptbxl_rhythm.

    python tools/p_morph/fetch_fresh.py --out data/ptbxl_fresh [--per-class 300]

Sinus tachycardia, normal sinus rhythm, first-degree AV block, AF and sinus arrhythmia / PACs: the classes a
P-shape SVT rule could hurt. Licensed data, git-ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
CLASSES = ("STACH", "NORM", "1AVB", "AFIB", "SARRH", "PAC")


def get(url: str) -> bytes:
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == 5:
                raise
            time.sleep(1.5 * (i + 1))
    return b""


def main() -> None:
    from cardiolens.validation.fetch_ptbxl import record_labels

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=271)
    a = ap.parse_args()
    used = set()
    for d in ("ptbxl", "ptbxl_rhythm"):
        used |= {r["ecg_id"] for r in json.loads((ROOT / "data" / d / "manifest.json").read_text())["records"]}
    rows = list(csv.DictReader((ROOT / "data/ptbxl/ptbxl_database.csv").open(newline="", encoding="utf-8")))
    rng = random.Random(a.seed)
    chosen: dict[int, dict] = {}
    for c in CLASSES:
        pool = [r for r in rows if int(float(r["ecg_id"])) not in used and int(float(r["ecg_id"])) not in chosen
                and c in record_labels(r)]
        for r in rng.sample(pool, min(a.per_class, len(pool))):
            chosen[int(float(r["ecg_id"]))] = r
    recs = [{"ecg_id": k, "filename_hr": r["filename_hr"], "labels": sorted(record_labels(r)),
             "age": r.get("age"), "sex": r.get("sex")} for k, r in chosen.items()]

    def dl(rec: dict) -> None:
        dest = a.out / rec["filename_hr"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            t = dest.with_suffix(ext)
            if not t.exists():
                t.write_bytes(get(BASE + rec["filename_hr"] + ext))

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(dl, recs))
    (a.out / "manifest.json").write_text(json.dumps({"source": BASE, "excludes": ["data/ptbxl", "data/ptbxl_rhythm"],
                                                     "records": recs}))
    print(f"[fresh] {len(recs)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
