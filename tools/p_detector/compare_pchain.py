"""D295: the current p_chain detector scored like the CNN (LUDB odd ids, lead II, +-40 ms; AF / flutter false P)."""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))


def pchain_peaks(x, rp, fs, qrs, qt):
    from scipy.signal import butter, filtfilt, find_peaks

    from cardiolens.layer3_measure.p_chain import P_CHAIN_BAND_HZ, P_CHAIN_MIN_UV, P_CHAIN_SD_K
    qrs = (qrs or 90) / 1000; qt = (qt or 400) / 1000; t_end = qt - qrs / 2 + 0.02
    bb, aa = butter(2, [P_CHAIN_BAND_HZ[0] / (fs / 2), P_CHAIN_BAND_HZ[1] / (fs / 2)], btype="band")
    y = filtfilt(bb, aa, x)
    wins = [(int(rp[i] + t_end * fs), int(rp[i + 1] - (qrs / 2 + 0.02) * fs)) for i in range(len(rp) - 1)]
    wins = [(a, b) for a, b in wins if b - a >= int(0.12 * fs)]
    if len(wins) < 3:
        return np.array([], int)
    dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
    h = max(P_CHAIN_SD_K * 1.4826 * float(np.median(np.abs(dia - np.median(dia)))), P_CHAIN_MIN_UV)
    best = []
    for pol in (1.0, -1.0):
        ps = []
        for a, b in wins:
            seg = pol * (y[a:b] - np.median(y[a:b]))
            pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(0.02 * fs), int(0.16 * fs)), distance=int(0.2 * fs))
            if len(pk):
                ps.append(a + int(pk[np.argmax(seg[pk])]))
        if len(ps) > len(best):
            best = ps
    return np.array(best, int)


SEL = (lambda i: i % 2 == 1) if 'val' not in sys.argv else (lambda i: i % 4 == 0)   # test = odd ids, val = ids % 4 == 0


def main():
    warnings.filterwarnings("ignore")
    import wfdb

    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    rows = {r["ID"]: r for r in csv.DictReader(open(ROOT / "data/ludb/ludb.csv"))}
    g = tp = fp = 0; af_d = n_af = 0; tol = 20
    for rid in sorted(rows, key=int):
        if not SEL(int(rid)):
            continue
        p = str(ROOT / "data/ludb/data" / rid)
        ann = wfdb.rdann(p, "ii"); sym = np.array(ann.symbol); smp = np.array(ann.sample)
        res = run_pipeline(load_wfdb(p), mode="A"); m = res.measurements; s = res.signal
        d = pchain_peaks(np.asarray(s.lead("II"), float), np.asarray(m.r_peaks_samples, int), s.sampling_rate_hz,
                         m.intervals.qrs_ms, m.intervals.qt_ms)
        d = d[(d >= smp.min()) & (d <= smp.max())] if len(d) else d
        if rows[rid]["Rhythms"].strip().startswith("Atrial"):
            af_d += len(d); n_af += 1; continue
        gold = smp[sym == "p"]
        g += len(gold); tp += int(sum(np.any(np.abs(d - x) <= tol) for x in gold)) if len(d) else 0
        fp += int(sum(not np.any(np.abs(gold - x) <= tol) for x in d)) if len(gold) else len(d)
    print(json.dumps({"recall": round(tp / g, 3), "precision": round(tp / max(1, tp + fp), 3), "af_false_per_record": round(af_d / max(1, n_af), 2)}))


if __name__ == "__main__":
    main()
