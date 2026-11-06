"""D205: AF feature rows for training (Chapman/Ningbo) and checking (PTB-XL bench).

    python tools/af_train/dump.py chapman --out <dir> [--shard k/n]
    python tools/af_train/dump.py ptbxl --data data/ptbxl_rhythm --out <dir> [--shard k/n]

One JSON per record: labels, the rule's codes, the AF feature dict (ml/af.py). Licensed data stays
in data/ (git-ignored); only these scripts and the trained coefficients go to git.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Chapman/Ningbo SNOMED (ConditionNames_SNOMED-CT.csv): AF and the rhythms it is confused with.
CHAPMAN = {"164889003": "AF", "164890007": "AFL", "284470004": "APB", "17338001": "VPB",
           "427393009": "SA", "426783006": "SR", "426177001": "SB", "427084000": "ST", "426761007": "SVT"}
PER_CLASS = {"AF": 1500, "AFL": 400, "APB": 600, "VPB": 500, "SA": 600, "SR": 500, "SB": 400, "ST": 400,
             "SVT": 300}
DIRS = ("data/ecg-arrhythmia-extra", "data/ecg-arrhythmia-train", "data/ecg-arrhythmia-balanced", "data/ecg-arrhythmia-pediatric",
        "data/ecg-arrhythmia")


def chapman_list(seed: int = 0, require_signal: bool = True) -> list[tuple[str, list[str], float | None]]:
    seen: dict[str, tuple[str, list[str], float | None]] = {}
    for d in DIRS:
        for hea in Path(d).rglob("*.hea"):
            # header-only mirrors (".headers/") have no signal next to them
            if hea.stem in seen or (require_signal and not hea.with_suffix(".mat").exists()):
                continue
            dx, age = [], None
            for line in hea.read_text(errors="ignore").splitlines():
                if line.startswith("#Dx:"):
                    dx = [c.strip() for c in line[4:].split(",")]
                elif line.startswith("#Age:"):
                    try:
                        age = float(line[5:].strip())
                    except ValueError:
                        age = None
            labels = sorted({CHAPMAN[c] for c in dx if c in CHAPMAN})
            if labels and (age is None or age >= 18):
                seen[hea.stem] = (str(hea.with_suffix("")), labels, age)
    rng = random.Random(seed)
    chosen: dict[str, tuple] = {}
    for cls, k in PER_CLASS.items():
        pool = [v for name, v in sorted(seen.items()) if cls in v[1] and name not in chosen]
        for v in rng.sample(pool, min(k, len(pool))):
            chosen[Path(v[0]).name] = v
    return list(chosen.values())


def run(items, out: Path, shard: str) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.ml.af import af_features
    from cardiolens.pipeline import run_pipeline

    k, n = (int(v) for v in shard.split("/"))
    out.mkdir(parents=True, exist_ok=True)
    for path, labels, age in items[k::n]:
        dest = out / (Path(path).name + ".json")
        if dest.exists():
            continue
        try:
            sig = load_wfdb(path)
            sig.patient.age_years = age
            res = run_pipeline(sig, mode="A")
            row = {"labels": labels, "codes": sorted(res.findings.codes()),
                   "f": af_features(res.signal, res.measurements)}
        except Exception as e:     # a failure is a row too
            row = {"labels": labels, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        dest.write_text(json.dumps(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["chapman", "ptbxl"])
    ap.add_argument("--data", type=Path, default=Path("data/ptbxl_rhythm"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.which == "chapman":
        # The same selection as extract.py; a record read from the zip lives in the extra folder.
        items = []
        for path, labels, age in chapman_list(require_signal=False):
            if not Path(path).with_suffix(".mat").exists():
                path = str(Path("data/ecg-arrhythmia-extra") / Path(path).name)
            items.append((path, labels, age))
    else:
        man = json.loads((a.data / "manifest.json").read_text())["records"]
        items = [(str(a.data / r["filename_hr"]), r["labels"],
                  float(r["age"]) if r.get("age") not in (None, "") else None) for r in man]
    run(items, a.out, a.shard)


if __name__ == "__main__":
    main()
