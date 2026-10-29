"""D182 step 1 (project venv): the records, labels, split and rhythm_hgb probabilities.

Same records and split as cardiolens.ml.train (D168): train = ecg-arrhythmia-train +
ecg-arrhythmia + PTB-XL folds 1-8; hold-outs = ecg-arrhythmia-balanced and PTB-XL
folds 9-10 (hold-out names removed from training). rhythm_hgb is scored on the cached
features of the hold-outs with the model file the site uses.

    python tools/fm_probe/export.py --out <dir outside git>/records.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cardiolens.ml.model import load_model
from cardiolens.ml.train import CLASSES

DATA = Path("data")


def files(root: Path) -> dict[str, str]:
    """Record name -> WFDB record path (no suffix)."""
    if (root / "manifest.json").exists() and (root / "ptbxl_database.csv").exists():
        recs = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["records"]
        return {f"ptbxl_{r['ecg_id']}": str(root / r["filename_hr"]) for r in recs}
    return {f"ecgarr_{h.stem}": str(h.with_suffix("")) for h in root.rglob("*.hea")
            if h.with_suffix(".mat").exists() and not any(p.startswith(".") for p in h.relative_to(root).parts)}


def ptbxl_folds(root: Path) -> dict[str, int]:
    with (root / "ptbxl_database.csv").open(encoding="utf-8") as f:
        return {f"ptbxl_{row['ecg_id']}": int(float(row["strat_fold"])) for row in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cardiolens/rhythm_hgb.joblib")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    model = load_model(a.model)
    if model is None:
        raise SystemExit(f"no model at {a.model}")
    folds = ptbxl_folds(DATA / "ptbxl")
    rows: dict[str, dict] = {}
    for name, split in (("ecg-arrhythmia-train", "train"), ("ecg-arrhythmia", "train"),
                        ("ecg-arrhythmia-balanced", "holdout_ecgarr"), ("ptbxl", None)):
        root = DATA / name
        z = np.load(root / ".ml_features.npz", allow_pickle=True)
        where = files(root)
        for x, gold, n in zip(z["x"], z["gold"], z["names"]):
            s = split or ("train" if folds.get(str(n), 0) <= 8 else "holdout_ptbxl")
            if str(n) not in where:
                continue
            row = {"name": str(n), "path": where[str(n)], "split": s, "gold": json.loads(str(gold))}
            if s.startswith("holdout"):
                row["hgb"] = model.predict_proba(x)
            prev = rows.get(str(n))
            if prev is None or (s.startswith("holdout") and not prev["split"].startswith("holdout")):
                rows[str(n)] = row       # a hold-out name never trains (cardiolens.ml.train)
    out = list(rows.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"classes": CLASSES, "model_classes": model.classes, "records": out}))
    for s in ("train", "holdout_ecgarr", "holdout_ptbxl"):
        print(s, sum(r["split"] == s for r in out))


if __name__ == "__main__":
    main()
