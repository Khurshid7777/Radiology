"""D185 real-photo bench: the PhysioNet-Kaggle "Digitization of ECG Images" training records
(Hugging Face mirror Emulated-Inc/ecgdig), nine images per record (v1 clean render ... v4 phone
photo, v5 photo of a laptop screen, v6 soaked, v7 damaged, v8/v9 mould scans), read through
the site's photo route and compared with the record's true signal.

    python tools/ecgdig_bench/run.py --data <dir with train.csv and train/> --out <dir> [--shard 0/3]

One JSON per image in <out>/rows (a stopped run resumes); ``--summary`` prints the table.
Metrics: read rate; HR within 5 bpm and QRS within 20 ms of the pipeline on the true signal;
"silent wrong" (HR off > 10 % with no photo gate); lead II correlation and SNR (dB) against
the truth after the best shift within +-1 s (the Challenge's measure, without its rescaling).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
WINDOWS = {"I": (0, 2.5), "II": (0, 10), "III": (0, 2.5), "aVR": (2.5, 5), "aVL": (2.5, 5), "aVF": (2.5, 5),
           "V1": (5, 7.5), "V2": (5, 7.5), "V3": (5, 7.5), "V4": (7.5, 10), "V5": (7.5, 10), "V6": (7.5, 10)}
GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible", "photo_coarse"}
CONDITIONS = {"v1": "clean render", "v2": "colour scan", "v3": "b/w scan", "v4": "phone photo",
              "v5": "photo of a screen", "v6": "soaked, photo", "v7": "damaged, photo",
              "v8": "mould, colour scan", "v9": "mould, b/w scan"}


def truth_signal(csv_path: Path, fs: float):
    from cardiolens.contracts.signal import Provenance, SignalRepresentation

    rows = list(csv.reader(csv_path.open()))
    head, body = rows[0], rows[1:]
    x = np.array([[float(v) if v else np.nan for v in r] for r in body], dtype=np.float64).T
    sig = np.zeros((12, x.shape[1]), dtype=np.float32)
    for i, lead in enumerate(LEADS):
        sig[i] = np.nan_to_num(x[head.index(lead)]) * 1000.0          # mV -> µV
    return SignalRepresentation(signal=sig, sampling_rate_hz=fs, lead_names=LEADS, duration_s=x.shape[1] / fs,
                                source="csv", provenance=Provenance(lead_windows=dict(WINDOWS))), x[head.index("II")]


def lead_ii_match(sig, truth_ii: np.ndarray, fs: float) -> tuple[float | None, float | None]:
    """Correlation and SNR (dB) of the read lead II against the truth, best shift within +-1 s."""
    from scipy import signal as sps

    if "II" not in sig.lead_names:
        return None, None
    est = np.asarray(sig.signal[sig.lead_names.index("II")], dtype=np.float64) / 1000.0
    est = sps.resample(est, int(round(len(est) * fs / sig.sampling_rate_hz)))
    t = truth_ii - np.nanmean(truth_ii)
    best = (-2.0, None)
    for lag in range(-int(fs), int(fs) + 1, max(1, int(fs // 100))):
        a = t[max(0, lag):len(t) + min(0, lag)]
        b = est[max(0, -lag):max(0, -lag) + len(a)]
        if len(b) < len(a) * 0.8 or np.std(b) == 0:
            continue
        a2 = a[:len(b)]
        b2 = b - np.mean(b)
        r = float(np.corrcoef(a2, b2)[0, 1])
        if r > best[0]:
            snr = 10 * np.log10(np.sum(a2 ** 2) / max(np.sum((a2 - b2) ** 2), 1e-12))
            best = (r, float(snr))
    return (round(best[0], 3), round(best[1], 2)) if best[1] is not None else (None, None)


def run(a) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    fs_of = {r["id"]: float(r["fs"]) for r in csv.DictReader((a.data / "train.csv").open())}
    recs = sorted(p.name for p in (a.data / "train").iterdir() if p.is_dir())
    k, n = (int(v) for v in a.shard.split("/"))
    rows = a.out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    for rec in recs[k::n]:
        fs = fs_of[rec]
        truth, truth_ii = truth_signal(a.data / "train" / rec / f"{rec}.csv", fs)
        mt = run_pipeline(truth, mode="A").measurements
        for img in sorted((a.data / "train" / rec).glob(f"{rec}-v*.*")):
            cond = img.stem.split("-")[1]
            out = rows / f"{rec}-{cond}.json"
            if out.exists():
                continue
            row = {"record": rec, "cond": cond, "hr_true": mt.heart_rate_bpm, "qrs_true": mt.intervals.qrs_ms}
            try:
                sig = digitize_photos([img])
                res = run_pipeline(sig, mode="A")
                corr, snr = lead_ii_match(sig, truth_ii, fs)
                row.update(read=True, hr=res.measurements.heart_rate_bpm, qrs=res.measurements.intervals.qrs_ms,
                           gated=bool(res.findings.codes() & GATES), corr_ii=corr, snr_ii=snr,
                           route="scanner" if sig.provenance.notes and sig.provenance.notes[0].startswith("ecgscan")
                           else "readers")
            except Exception as e:  # a refusal is a result too
                row.update(read=False, error=str(e)[:200])
            out.write_text(json.dumps(row, ensure_ascii=False))
            print(f"{rec} {cond} {row.get('hr')} / {row['hr_true']} gated={row.get('gated')} r={row.get('corr_ii')}",
                  file=sys.stderr, flush=True)


def summary(out: Path) -> dict:
    by = defaultdict(list)
    for f in sorted((out / "rows").glob("*.json")):
        r = json.loads(f.read_text())
        by[r["cond"]].append(r)
    table = {}
    for cond in sorted(by):
        rs = by[cond]
        read = [r for r in rs if r.get("read")]

        def ok(r, k, tol):
            return r.get(k) is not None and r.get(k + "_true") and abs(r[k] - r[k + "_true"]) <= tol
        silent = [r for r in read if not r["gated"] and r["hr"] and r["hr_true"]
                  and abs(r["hr"] - r["hr_true"]) > 0.1 * r["hr_true"]]
        corr = [r["corr_ii"] for r in read if r.get("corr_ii") is not None]
        snr = [r["snr_ii"] for r in read if r.get("snr_ii") is not None]
        table[cond] = {"what": CONDITIONS.get(cond, cond), "n": len(rs), "read": len(read),
                       "hr_ok": sum(ok(r, "hr", 5) for r in read), "qrs_ok": sum(ok(r, "qrs", 20) for r in read),
                       "gated": sum(r["gated"] for r in read), "silent_wrong": len(silent),
                       "corr_ii_median": round(float(np.median(corr)), 3) if corr else None,
                       "snr_ii_median": round(float(np.median(snr)), 2) if snr else None}
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        t = summary(a.out)
        (a.out / "summary.json").write_text(json.dumps(t, indent=1, ensure_ascii=False))
        for c, v in t.items():
            print(c, json.dumps(v, ensure_ascii=False))
        return
    run(a)


if __name__ == "__main__":
    main()
