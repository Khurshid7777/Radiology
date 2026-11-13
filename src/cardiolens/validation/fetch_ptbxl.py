"""Fetch a label-balanced subset of PTB-XL (PhysioNet) for validation (TZ §15, §16).

    python -m cardiolens.validation.fetch_ptbxl --out data/ptbxl --per-class 80 --seed 0

PTB-XL 1.0.3: ``ptbxl_database.csv`` (one row per record; ``scp_codes`` is a dict
SCP-code -> likelihood, ``heart_axis``, ``age``, ``sex``, ``filename_hr``) and the
500 Hz WFDB records under ``records500/``. The selection takes **every** record of
the rare classes (complete/second-degree AV block, flutter, WPW, long QT, non-specific
ST elevation, low voltage) and every child (age < 16), then fills the other classes up
to ``--per-class`` with ``select_balanced`` (rarest first, unlabeled negatives kept).
Only the selected ``.hea``/``.dat`` pairs are downloaded (keep-alive client from
``fetch_physionet``); ``manifest.json`` records the selection. Licensed data — never
committed (``data/ptbxl/`` is git-ignored).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cardiolens.validation.fetch_physionet import UNLABELED, fetch_bytes, select_balanced

# physionet.org answers 302 to its S3 mirror; the keep-alive client does not follow redirects.
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
LIKELIHOOD_MIN = 50.0   # scp likelihood accepted as a label (0 = not graded -> kept)
PEDIATRIC_AGE_MAX = 15.0
# Every record of these classes is taken (they are rare).
RARE_FULL = {"3AVB", "2AVB", "AFLT", "WPW", "LNGQT", "STE_", "LVOLT"}
# Classes that take part in the balanced fill (the rest of the SCP vocabulary is
# carried along in the manifest but does not drive selection).
BALANCE_CLASSES = {
    "SR", "AFIB", "STACH", "SBRAD", "SARRH", "1AVB", "LPR", "CRBBB", "CLBBB", "IRBBB",
    "ILBBB", "LAFB", "LPFB", "LVH", "VCLVH", "STD_", "IMI", "ASMI", "AMI", "ALMI", "ILMI",
    "LAO/LAE", "RAO/RAE", "PVC", "PAC", "NDT", "NST_", "ISC_", "AXIS_ALAD", "AXIS_RAD", "AXIS_ARAD",
}


def record_labels(row: dict) -> set[str]:
    """SCP codes with likelihood >= LIKELIHOOD_MIN (or ungraded) + axis pseudo-labels."""
    codes = ast.literal_eval(row["scp_codes"])
    labels = {k for k, v in codes.items() if float(v) >= LIKELIHOOD_MIN or float(v) == 0}
    axis = row.get("heart_axis")
    axis = axis.strip() if isinstance(axis, str) else ""
    # ALAD / RAD / ARAD are axis deviations by the +-30/90 criteria; PTB-XL's
    # "LAD" is the mild leftward axis (-30..0) and is not labelled (D49).
    if axis in ("ALAD", "RAD", "ARAD"):
        labels.add(f"AXIS_{axis}")
    return labels


def _age(row: dict) -> float | None:
    try:
        return float(row["age"])
    except (TypeError, ValueError):
        return None


def select_records(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    """Rare classes and children in full, the rest balanced (deterministic)."""
    forced, pool = [], {}
    for row in rows:
        labels = record_labels(row)
        age = _age(row)
        if labels & RARE_FULL or (age is not None and age <= PEDIATRIC_AGE_MAX):
            forced.append(row)
        else:
            pool[row["ecg_id"]] = labels & BALANCE_CLASSES
    chosen = set(select_balanced(pool, per_class, seed))
    by_id = {row["ecg_id"]: row for row in rows}
    return forced + [by_id[i] for i in sorted(chosen, key=int)]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch a balanced PTB-XL subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        dest = out / name
        if not dest.exists():
            dest.write_bytes(fetch_bytes(BASE_URL + name))
    with (out / "ptbxl_database.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chosen = select_records(rows, args.per_class, args.seed)
    support = Counter(c for r in chosen for c in (record_labels(r) or {UNLABELED}))
    _log(f"[ptbxl] selected {len(chosen)} of {len(rows)} records; "
         f"rare: {{{', '.join(f'{k}:{support.get(k, 0)}' for k in sorted(RARE_FULL))}}}; "
         f"children: {sum(1 for r in chosen if (_age(r) or 99) <= PEDIATRIC_AGE_MAX)}")

    def fetch_record(row: dict) -> str:
        rel = row["filename_hr"]                      # e.g. records500/00000/00001_hr
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            target = dest.with_suffix(ext)
            if not target.exists():
                target.write_bytes(fetch_bytes(BASE_URL + rel + ext))
        return row["ecg_id"]

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(fetch_record, chosen), 1):
            if i % 100 == 0 or i == len(chosen):
                _log(f"[ptbxl] records {i}/{len(chosen)}")

    manifest = {
        "source": BASE_URL, "seed": args.seed, "per_class": args.per_class,
        "records": [{
            "ecg_id": int(r["ecg_id"]), "filename_hr": r["filename_hr"],
            "age": _age(r), "sex": r.get("sex"), "heart_axis": r.get("heart_axis"),
            "scp_codes": ast.literal_eval(r["scp_codes"]), "labels": sorted(record_labels(r)),
        } for r in chosen],
        "support": dict(support),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log(f"[ptbxl] done -> {out}")


if __name__ == "__main__":
    main()
