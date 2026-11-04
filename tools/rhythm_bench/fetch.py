"""D201: the rhythm and conduction bench — a PTB-XL 1.0.3 subset aimed at tachyarrhythmias and blocks.

    python tools/rhythm_bench/fetch.py --out data/ptbxl_rhythm [--per-class 200] [--controls 400]

Every record of the rare classes, a seeded sample of the common ones, and normal sinus rhythm as the
specificity control. Labels are SCP codes with likelihood >= 50 (or ungraded), as in fetch_ptbxl.
Licensed data: the folder is git-ignored; only scripts and summary tables go to git.
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
from cardiolens.validation.fetch_ptbxl import record_labels

# physionet.org answers 302 to its S3 mirror; the keep-alive client does not follow redirects.
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def fetch_bytes(url: str, tries: int = 5) -> bytes:
    """Plain urllib (honours the proxy environment); the project's keep-alive client is pinned to one host."""
    import time
    import urllib.request

    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(url)

RARE_FULL = ("2AVB", "3AVB", "AFLT", "WPW", "SVTAC", "PSVT", "LPFB")
SAMPLED = ("LAFB", "IRBBB", "CRBBB", "CLBBB", "AFIB", "PVC", "PACE", "1AVB", "IVCD", "STACH")
# Sampled after the controls so the earlier selection stays the same (D203: a slow regular rhythm
# is where a 2:1 AV block hides; sinus bradycardia is its specificity control).
EXTRA = ("SBRAD",)


def select(rows: list[dict], per_class: int, controls: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: dict[str, dict] = {}
    labels = {r["ecg_id"]: record_labels(r) for r in rows}
    for code in RARE_FULL:
        for r in rows:
            if code in labels[r["ecg_id"]]:
                chosen[r["ecg_id"]] = r
    for code in SAMPLED:
        pool = [r for r in rows if code in labels[r["ecg_id"]] and r["ecg_id"] not in chosen]
        for r in rng.sample(pool, min(per_class, len(pool))):
            chosen[r["ecg_id"]] = r
    # Controls: normal sinus rhythm with nothing else of interest.
    interest = set(RARE_FULL) | set(SAMPLED)
    norm = [r for r in rows if {"NORM", "SR"} <= labels[r["ecg_id"]] and not labels[r["ecg_id"]] & interest
            and r["ecg_id"] not in chosen]
    for r in rng.sample(norm, min(controls, len(norm))):
        chosen[r["ecg_id"]] = r
    for code in EXTRA:
        pool = [r for r in rows if code in labels[r["ecg_id"]] and r["ecg_id"] not in chosen]
        for r in rng.sample(pool, min(per_class, len(pool))):
            chosen[r["ecg_id"]] = r
    return list(chosen.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=Path("data/ptbxl/ptbxl_database.csv"))
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--controls", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(a.db.open(encoding="utf-8")))
    chosen = select(rows, a.per_class, a.controls, a.seed)
    print(f"[rhythm] {len(chosen)} records", file=sys.stderr, flush=True)

    def get(row: dict) -> None:
        rel = row["filename_hr"]
        dest = a.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            t = dest.with_suffix(ext)
            if not t.exists():
                t.write_bytes(fetch_bytes(BASE_URL + rel + ext))

    with ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(get, chosen), 1):
            if i % 200 == 0 or i == len(chosen):
                print(f"[rhythm] {i}/{len(chosen)}", file=sys.stderr, flush=True)
    manifest = {"source": BASE_URL, "seed": a.seed, "records": [
        {"ecg_id": int(r["ecg_id"]), "filename_hr": r["filename_hr"], "age": r.get("age"), "sex": r.get("sex"),
         "heart_axis": r.get("heart_axis"), "scp_codes": ast.literal_eval(r["scp_codes"]),
         "labels": sorted(record_labels(r))} for r in chosen]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
