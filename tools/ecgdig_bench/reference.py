"""D185 step 2: score a reference digitiser's WFDB output (ECG-Digitiser, Challenge 2024) on the
ecgdig bench images with the same metrics as ``run.py`` — lead II correlation/SNR against the
truth, and HR/QRS from our pipeline run on the reference's signal (gates apply as for a photo).

    python tools/ecgdig_bench/reference.py --data <dir with train.csv and train/> --wfdb <dir> --out <dir>

One JSON per image in <out>/ref_rows; ``--summary`` prints the table next to ours (<out>/rows).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import CONDITIONS, GATES, LEADS, WINDOWS, lead_ii_match, truth_signal


def reference_signal(hea: Path):
    import wfdb

    from cardiolens.contracts.signal import Provenance, SignalRepresentation

    r = wfdb.rdrecord(str(hea.with_suffix("")))
    x = np.nan_to_num(r.p_signal.T.astype(np.float64)) * 1000.0                     # mV -> µV
    # a lead the reference failed to read is absent from its record: left flat
    sig = np.stack([x[r.sig_name.index(lead)] if lead in r.sig_name else np.zeros(x.shape[1])
                    for lead in LEADS]).astype(np.float32)
    return SignalRepresentation(signal=sig, sampling_rate_hz=float(r.fs), lead_names=LEADS,
                                duration_s=sig.shape[1] / r.fs, source="image",
                                provenance=Provenance(digitized=True, lead_windows=dict(WINDOWS)))


def run(a) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.pipeline import run_pipeline

    fs_of = {r["id"]: float(r["fs"]) for r in csv.DictReader((a.data / "train.csv").open())}
    rows = a.out / "ref_rows"
    rows.mkdir(parents=True, exist_ok=True)
    for hea in sorted(a.wfdb.glob("*.hea")):
        rec, cond = hea.stem.split("-")
        out = rows / f"{hea.stem}.json"
        if out.exists():
            continue
        fs = fs_of[rec]
        truth, truth_ii = truth_signal(a.data / "train" / rec / f"{rec}.csv", fs)
        mt = run_pipeline(truth, mode="A").measurements
        sig = reference_signal(hea)
        res = run_pipeline(sig, mode="A")
        corr, snr = lead_ii_match(sig, truth_ii, fs)
        row = {"record": rec, "cond": cond, "hr_true": mt.heart_rate_bpm, "qrs_true": mt.intervals.qrs_ms,
               "read": True, "hr": res.measurements.heart_rate_bpm, "qrs": res.measurements.intervals.qrs_ms,
               "gated": bool(res.findings.codes() & GATES), "corr_ii": corr, "snr_ii": snr}
        out.write_text(json.dumps(row, ensure_ascii=False))
        print(f"{rec} {cond} {row['hr']} / {row['hr_true']} r={corr}", file=sys.stderr, flush=True)


def summary(out: Path) -> dict:
    """Per condition, on the images the reference was run on: the reference against ours."""
    ref = {p.stem: json.loads(p.read_text()) for p in (out / "ref_rows").glob("*.json")}
    ours = {p.stem: json.loads(p.read_text()) for p in (out / "rows").glob("*.json")}
    table = {}
    for cond, what in CONDITIONS.items():
        keys = sorted(k for k in ref if k.endswith(f"-{cond}"))
        if not keys:
            continue

        def stats(rows):
            read = [r for r in rows if r.get("read")]
            ok = [r for r in read if r.get("hr") and r.get("hr_true") and abs(r["hr"] - r["hr_true"]) <= 5]
            corr = [r["corr_ii"] for r in read if r.get("corr_ii") is not None]
            snr = [r["snr_ii"] for r in read if r.get("snr_ii") is not None]
            return {"read": len(read), "hr_ok": len(ok),
                    "gated": sum(bool(r.get("gated")) for r in read),
                    "corr_ii_median": round(float(np.median(corr)), 3) if corr else None,
                    "snr_ii_median": round(float(np.median(snr)), 2) if snr else None}
        table[cond] = {"what": what, "n": len(keys), "reference": stats([ref[k] for k in keys]),
                       "ours": stats([ours[k] for k in keys if k in ours])}
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--wfdb", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(json.dumps(summary(a.out), indent=1, ensure_ascii=False))
    else:
        run(a)


if __name__ == "__main__":
    main()
