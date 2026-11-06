"""D201: run the pipeline over the rhythm and conduction bench and score each class.

    python tools/rhythm_bench/run.py --data data/ptbxl_rhythm --out <dir> [--shard k/n] [--meas]   # rows, resumable
    python tools/rhythm_bench/run.py --data data/ptbxl_rhythm --out <dir> --score [--md table.md]

A row per record (codes found, rate, QRS, PR): <out>/rows/<ecg_id>.json. Scoring: for each PTB-XL class
its CardioLens code(s); sensitivity on records carrying the class, specificity on the normal sinus
controls (NORM + SR, nothing of interest), and the number of controls the code fired on.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# PTB-XL class -> the CardioLens codes that count as finding it (a detector may be stricter or more
# general; e.g. a high-grade block found as complete counts for 2AVB under "do not miss").
CLASS_CODES = {
    "3AVB": {"complete_av_block", "slow_wide_escape_rhythm"},
    "2AVB": {"second_degree_av_block", "av_block_2_1", "high_grade_av_block", "complete_av_block"},
    "1AVB": {"first_degree_av_block"},
    "AFIB": {"atrial_fibrillation", "possible_atrial_fibrillation"},
    "AFLT": {"atrial_flutter"},
    "SVTAC": {"supraventricular_tachycardia", "atrial_tachycardia"},
    "PSVT": {"supraventricular_tachycardia", "atrial_tachycardia"},
    "STACH": {"sinus_tachycardia"},
    "WPW": {"wpw_preexcitation"},
    "CLBBB": {"lbbb"},
    "CRBBB": {"rbbb"},
    "IRBBB": {"incomplete_rbbb"},
    "LAFB": {"left_anterior_fascicular_block"},
    "LPFB": {"left_posterior_fascicular_block"},
    "IVCD": {"wide_qrs_nonspecific"},
    "PVC": {"ventricular_ectopy", "ventricular_bigeminy", "ventricular_trigeminy"},
    "PACE": {"paced_rhythm"},
    # D204 wave 1
    "PAC": {"supraventricular_ectopy", "atrial_bigeminy", "atrial_trigeminy"},
    "SVARR": {"supraventricular_ectopy", "atrial_bigeminy", "atrial_trigeminy"},
    "BIGU": {"ventricular_bigeminy"},
    "TRIGU": {"ventricular_trigeminy"},
    "SARRH": {"sinus_arrhythmia"},
    "ILBBB": {"incomplete_lbbb"},
}
INTEREST = set(CLASS_CODES)


def run_rows(data: Path, out: Path, shard: str, meas: bool = False) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline

    man = json.loads((data / "manifest.json").read_text())["records"]
    k, n = (int(v) for v in shard.split("/"))
    rows = out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    mdir = out / "meas"
    if meas:
        mdir.mkdir(exist_ok=True)
    for rec in man[k::n]:
        dest = rows / f"{rec['ecg_id']}.json"
        mdest = mdir / f"{rec['ecg_id']}.json"
        if dest.exists() and (not meas or mdest.exists()):
            continue
        row = {"ecg_id": rec["ecg_id"], "labels": rec["labels"]}
        try:
            sig = load_wfdb(str(data / rec["filename_hr"]))
            try:
                sig.patient.age_years = float(rec["age"]) if rec.get("age") not in (None, "") else None
            except ValueError:
                pass
            sex = str(rec.get("sex"))
            sig.patient.sex = {"0": "male", "1": "female", "0.0": "male", "1.0": "female"}.get(sex)
            res = run_pipeline(sig, mode="A")
            m = res.measurements
            row.update(codes=sorted(res.findings.codes()), hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms,
                       pr=m.intervals.pr_ms, axis=m.axes.qrs_deg,
                       alerts=[a.code for a in res.safety.alerts])
            if meas:   # the measurements, to re-run a rule without the pipeline while tuning it
                mdest.write_text(json.dumps({"labels": rec["labels"], "age": sig.patient.age_years,
                                             "sex": sig.patient.sex, "m": m.model_dump(mode="json")}))
        except Exception as e:  # a failure is a result too
            row.update(error=f"{type(e).__name__}: {str(e)[:120]}")
        dest.write_text(json.dumps(row))
        print(rec["ecg_id"], row.get("codes", row.get("error")), file=sys.stderr, flush=True)


def is_control(labels: set[str]) -> bool:
    return {"NORM", "SR"} <= labels and not labels & INTEREST


def score(out: Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    rows = [r for r in rows if "codes" in r]
    controls = [r for r in rows if is_control(set(r["labels"]))]
    table = []
    for cls, codes in CLASS_CODES.items():
        pos = [r for r in rows if cls in r["labels"]]
        if not pos:
            continue
        tp = sum(bool(set(r["codes"]) & codes) for r in pos)
        fp = sum(bool(set(r["codes"]) & codes) for r in controls)
        table.append({"class": cls, "codes": "/".join(sorted(codes)), "n": len(pos), "found": tp,
                      "sens": round(tp / len(pos), 3),
                      "spec_ctrl": round(1 - fp / len(controls), 3) if controls else None,
                      "fp_ctrl": fp})
    table.append({"class": "controls", "n": len(controls), "errors": sum(1 for p in (out / "rows").glob("*.json")
                                                                       if "error" in json.loads(p.read_text()))})
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--meas", action="store_true", help="also keep each record's measurements (tuning)")
    a = ap.parse_args()
    if not a.score:
        run_rows(a.data, a.out, a.shard, a.meas)
        return
    t = score(a.out)
    lines = ["| class | n | found | sens | spec on controls | false on controls |", "|---|---|---|---|---|---|"]
    for r in t[:-1]:
        lines.append(f"| {r['class']} | {r['n']} | {r['found']} | {r['sens']} | {r['spec_ctrl']} | {r['fp_ctrl']} |")
    lines.append(f"\ncontrols: {t[-1]['n']}, errors: {t[-1]['errors']}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
