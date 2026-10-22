"""Score findings against reference labels through one mapping table (D102).

``configs/statement_map.yaml`` maps every finding code to SNOMED CT (PhysioNet Challenge
2020/2021 labels) and PTB-XL SCP statements. Codes sharing the same reference labels
(``lvh`` / ``lvh_probable``, ``st_elevation`` / ``st_elevation_any``) are scored as one
group: the group is predicted when any of its codes is. Sensitivity and specificity come
with Wilson 95 % intervals.

CLI::

    python -m cardiolens.validation.statement_score RUN.csv --ptbxl ptbxl_database.csv
    python -m cardiolens.validation.statement_score RUN.csv --challenge ROOT LIST.txt
    python -m cardiolens.validation.statement_score RUN.csv --code15 exams.csv

``RUN.csv`` has ``ecg_id`` and space-separated ``codes`` columns; ``LIST.txt`` lines are
``<id> <record path under ROOT>``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path

import yaml

from cardiolens.config import configs_dir
from cardiolens.validation.metrics import ClassMetrics, score_findings


def load_statement_map(path: Path | None = None) -> dict[str, dict]:
    return yaml.safe_load((path or configs_dir() / "statement_map.yaml").read_text())


def reference_groups(table: dict[str, dict], system: str) -> dict[str, tuple[set[str], set[str]]]:
    """``{group name: (finding codes, reference labels)}`` for one label system."""
    groups: dict[frozenset, list[str]] = {}
    for code, row in table.items():
        labels = frozenset(str(x) for x in row.get(system) or [])
        if labels:
            groups.setdefault(labels, []).append(code)
    return {codes[0]: (set(codes), set(labels)) for labels, codes in groups.items()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def score(predicted: dict[str, set[str]], labels: dict[str, set[str]], system: str,
          table: dict[str, dict] | None = None) -> dict[str, ClassMetrics]:
    """Per-group confusion counts over the records present in both mappings."""
    groups = reference_groups(table or load_statement_map(), system)
    ids = sorted(set(predicted) & set(labels))
    preds = [{g for g, (codes, _) in groups.items() if predicted[i] & codes} for i in ids]
    gold = [{g for g, (_, ref) in groups.items() if labels[i] & ref} for i in ids]
    return score_findings(preds, gold, sorted(groups))


def _ptbxl_labels(db: Path) -> dict[str, set[str]]:
    return {str(int(float(r["ecg_id"]))): set(ast.literal_eval(r["scp_codes"]))
            for r in csv.DictReader(db.open())}


def _challenge_labels(root: Path, listing: Path) -> dict[str, set[str]]:
    out = {}
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        rid, path = line.split()[:2]
        for h in (root / f"{path}.hea").read_text().splitlines():
            if h.startswith("# Dx:"):
                out[rid] = {x.strip() for x in h.split(":", 1)[1].split(",") if x.strip()}
    return out


_CODE15_COLUMNS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")


def _code15_labels(exams: Path) -> dict[str, set[str]]:
    """CODE-15% ``exams.csv``: one boolean column per label; ids are ``exam_id``."""
    return {str(r["exam_id"]): {c for c in _CODE15_COLUMNS if r.get(c) == "True"}
            for r in csv.DictReader(exams.open())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ptbxl", type=Path)
    src.add_argument("--challenge", nargs=2, type=Path, metavar=("ROOT", "LIST"))
    src.add_argument("--code15", type=Path, metavar="EXAMS_CSV")
    a = ap.parse_args(argv)
    if a.ptbxl:
        labels, system = _ptbxl_labels(a.ptbxl), "scp"
    elif a.code15:
        labels, system = _code15_labels(a.code15), "code15"
    else:
        labels, system = _challenge_labels(*a.challenge), "snomed"
    predicted = {r["ecg_id"]: set((r.get("codes") or "").split()) for r in csv.DictReader(a.run.open())}
    res = score(predicted, labels, system)
    print(f"records {len(set(predicted) & set(labels))} ({system})")
    print("| group | found | sens (95% CI) | false | spec (95% CI) |")
    print("|---|---|---|---|---|")
    for g, m in sorted(res.items(), key=lambda kv: -kv[1].support):
        if m.support == 0 and m.fp == 0:
            continue
        se, sp = wilson(m.tp, m.tp + m.fn), wilson(m.tn, m.tn + m.fp)
        print(f"| {g} | {m.tp}/{m.support} | {m.sensitivity} {se} | {m.fp} | {m.specificity} {sp} |")


if __name__ == "__main__":
    main()
