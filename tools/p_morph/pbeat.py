"""D276: per-beat P in diastole, with sign, per lead.

    python tools/p_morph/pbeat.py chapman|fresh2 --out rows.jsonl [--shard k/n] [--parity even|odd|all]

Same diastole search as p_chain (1-25 Hz, T end .. next QRS onset, a P-like wave above 2 robust SD), but the
two polarities are reported separately per lead: share of diastoles with a positive P, with a negative P, and
the spread of the P->next-R interval. Rows carry only numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
LEADS = ("II", "III", "aVF", "I", "aVR", "V1")


def beat_p(x, rp, fs, qrs_ms, qt_ms):
    qrs = (qrs_ms or 90.0) / 1000.0
    qt = (qt_ms or 400.0) / 1000.0
    t_end = qt - qrs / 2 + 0.02
    nyq = fs / 2.0
    bb, aa = butter(2, [1.0 / nyq, min(25.0 / nyq, 0.99)], btype="band")
    y = filtfilt(bb, aa, np.asarray(x, float))
    wmax = int(0.16 * fs)
    wins = [(int(rp[i] + t_end * fs), int(rp[i + 1] - (qrs / 2 + 0.02) * fs)) for i in range(len(rp) - 1)]
    wins = [(a, b) for a, b in wins if b - a >= int(0.12 * fs) and a >= 0 and b <= len(y)]
    if len(wins) < 3:
        return None
    dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
    h = max(2.0 * 1.4826 * float(np.median(np.abs(dia - np.median(dia)))), 15.0)
    out = {"n": len(wins)}
    for tag, pol in (("pos", 1.0), ("neg", -1.0)):
        cv, pr = 0, []
        for i, (a, b) in enumerate(wins):
            seg = pol * (y[a:b] - np.median(y[a:b]))
            pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(0.02 * fs), wmax), distance=int(0.2 * fs))
            if len(pk):
                cv += 1
                j = a + int(pk[np.argmax(seg[pk])])
                nxt = rp[[k for k in range(len(rp) - 1) if int(rp[k] + t_end * fs) == a][0] + 1]
                pr.append((nxt - j) / fs * 1000.0)
        out[tag] = round(cv / len(wins), 3)
        out[f"{tag}_prsd"] = round(float(np.std(pr)), 1) if len(pr) >= 3 else None
        out[f"{tag}_pr"] = round(float(np.median(pr)), 0) if pr else None
    return out


def row_for(sig, labels, ident):
    from cardiolens.pipeline import run_pipeline
    res = run_pipeline(sig, mode="A")
    m, s = res.measurements, res.signal
    rp = np.asarray(m.r_peaks_samples or [], int)
    rr = np.diff(rp)
    row = {"id": ident, "labels": labels, "codes": sorted(res.findings.codes()), "hr": m.heart_rate_bpm,
           "qrs": m.intervals.qrs_ms, "pr": m.intervals.pr_ms,
           "cv": float(rr.std() / rr.mean()) if rr.size > 2 else None}
    if rp.size >= 4:
        for ln in LEADS:
            if ln in s.lead_names:
                b = beat_p(s.lead(ln), rp, s.sampling_rate_hz, m.intervals.qrs_ms, m.intervals.qt_ms)
                if b:
                    row[ln] = b
    return row


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["chapman", "fresh2"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--parity", default="all", choices=["even", "odd", "all"])
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    if a.which == "chapman":
        recs = json.loads((ROOT / "data/svt_bench/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/svt_bench" / r["folder"] / r["name"]), r["labels"], r["name"], r.get("age")) for r in recs]
        par = lambda ident: int(str(ident)[2:]) % 2
    else:
        recs = json.loads((ROOT / "data/ptbxl_fresh2/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/ptbxl_fresh2" / r["filename_hr"]), r["labels"], r["ecg_id"], r.get("age")) for r in recs]
        par = lambda ident: int(ident) % 2
    if a.parity != "all":
        items = [it for it in items if par(it[2]) == (0 if a.parity == "even" else 1)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for path, labels, ident, age in items[k::n]:
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                fh.write(json.dumps(row_for(sig, labels, ident)) + "\n")
            except Exception as ex:
                fh.write(json.dumps({"id": ident, "labels": labels, "error": f"{type(ex).__name__}: {str(ex)[:60]}"}) + "\n")


if __name__ == "__main__":
    main()
