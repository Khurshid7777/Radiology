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


AVB_CODES = {"27885002": "3AVB", "195042002": "2AVB", "54016002": "MOBITZ1", "270492004": "1AVB",
             "233917008": "AVB", "426177001": "SB", "284470004": "APB", "426783006": "SR"}
AVB_PER_CLASS = {"3AVB": 200, "2AVB": 200, "MOBITZ1": 200, "AVB": 150, "1AVB": 300, "SB": 300, "APB": 300,
                 "SR": 200}


# D222: escape and junctional beats (Chapman labels are per record) and their controls.
ESC_CODES = {"426995002": "JEB", "75532003": "VEB", "251180001": "VET", "251164006": "JPT",
             "426177001": "SB", "426783006": "SR", "284470004": "APB", "427172004": "VPB"}
ESC_PER_CLASS = {"JEB": 200, "VEB": 200, "VET": 50, "JPT": 100, "SB": 300, "SR": 300, "APB": 200, "VPB": 200}


# D230: atrial tachycardia and the rhythms it is confused with.
AT_CODES = {"713422000": "AT", "427084000": "ST", "426761007": "SVT", "233896004": "AVNRT",
            "233897008": "AVRT", "164890007": "AFL"}
AT_PER_CLASS = {"AT": 400, "ST": 400, "SVT": 300, "AVNRT": 50, "AVRT": 50, "AFL": 200}


# D234: pre-excitation and the bundle branch blocks it is confused with.
WPW_CODES = {"74390002": "WPW", "164909002": "LBBB", "733534002": "CLBBB", "59118001": "RBBB",
             "713427006": "CRBBB", "426783006": "SR"}
WPW_PER_CLASS = {"WPW": 300, "LBBB": 300, "CLBBB": 300, "RBBB": 200, "CRBBB": 200, "SR": 300}


def chapman_list(seed: int = 0, require_signal: bool = True, codes: dict | None = None,
                 per_class: dict | None = None) -> list[tuple[str, list[str], float | None]]:
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
            labels = sorted({(codes or CHAPMAN)[c] for c in dx if c in (codes or CHAPMAN)})
            if labels and (age is None or age >= 18):
                seen[hea.stem] = (str(hea.with_suffix("")), labels, age)
    rng = random.Random(seed)
    chosen: dict[str, tuple] = {}
    for cls, k in (per_class or PER_CLASS).items():
        pool = [v for name, v in sorted(seen.items()) if cls in v[1] and name not in chosen]
        for v in rng.sample(pool, min(k, len(pool))):
            chosen[Path(v[0]).name] = v
    return list(chosen.values())


def run(items, out: Path, shard: str, keep_meas: bool = False) -> None:
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
            if keep_meas:   # D206: the measurements and the P train, to tune the AV-block rule offline
                import numpy as np

                from cardiolens.layer3_measure.atrial import P_TRAIN_LEADS, p_train

                row["m"] = res.measurements.model_dump(mode="json", exclude={"morphology", "voltages"})
                sg = res.signal
                row["p_train"] = p_train({ln: sg.lead(ln).astype(np.float64) for ln in P_TRAIN_LEADS
                                          if ln in sg.lead_names}, np.asarray(res.measurements.r_peaks_samples),
                                         sg.sampling_rate_hz)
        except Exception as e:     # a failure is a row too
            row = {"labels": labels, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        dest.write_text(json.dumps(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["chapman", "chapman-avb", "chapman-esc", "chapman-at", "chapman-wpw", "ptbxl"])
    ap.add_argument("--data", type=Path, default=Path("data/ptbxl_rhythm"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.which in ("chapman", "chapman-avb", "chapman-esc", "chapman-at", "chapman-wpw"):
        # The same selection as extract.py; a record read from the zip lives in the extra folder.
        kw = ({"codes": AVB_CODES, "per_class": AVB_PER_CLASS} if a.which == "chapman-avb" else
              {"codes": ESC_CODES, "per_class": ESC_PER_CLASS} if a.which == "chapman-esc" else
              {"codes": AT_CODES, "per_class": AT_PER_CLASS} if a.which == "chapman-at" else
              {"codes": WPW_CODES, "per_class": WPW_PER_CLASS} if a.which == "chapman-wpw" else {})
        items = []
        for path, labels, age in chapman_list(require_signal=False, **kw):
            if not Path(path).with_suffix(".mat").exists():
                path = str(Path("data/ecg-arrhythmia-extra") / Path(path).name)
            items.append((path, labels, age))
    else:
        man = json.loads((a.data / "manifest.json").read_text())["records"]
        items = [(str(a.data / r["filename_hr"]), r["labels"],
                  float(r["age"]) if r.get("age") not in (None, "") else None) for r in man]
    run(items, a.out, a.shard, keep_meas=a.which in ("chapman-avb", "chapman-esc", "chapman-at", "chapman-wpw"))


if __name__ == "__main__":
    main()
