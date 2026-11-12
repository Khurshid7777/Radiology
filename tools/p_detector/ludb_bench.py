"""D249: P-wave detection vs LUDB beat annotations (lead II): recall / precision of P peaks per setting, false P
on AF / flutter records; even / odd record ids. Needs data/ludb (PhysioNet LUDB 1.0.1, git-ignored)."""
import sys, csv, warnings, itertools; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

from cardiolens.layer1_ingestion import load_wfdb
from cardiolens.pipeline import run_pipeline

rows = {r["ID"]: r for r in csv.DictReader(open("data/ludb/ludb.csv"))}
cache = []
for rid in sorted(rows, key=int):
    ann = wfdb.rdann(f"data/ludb/data/{rid}", "ii"); sym, smp = np.array(ann.symbol), np.array(ann.sample)
    sig = load_wfdb(f"data/ludb/data/{rid}"); res = run_pipeline(sig, mode="A"); m = res.measurements; s = res.signal
    cache.append(dict(id=int(rid), rhythm=rows[rid]["Rhythms"].strip().split("\n")[0], gold=smp[sym == "p"], lo=smp.min() - 50, hi=smp.max() + 50,
                      x=np.asarray(s.lead("II"), float), rp=np.asarray(m.r_peaks_samples, int), fs=s.sampling_rate_hz,
                      qrs=m.intervals.qrs_ms, qt=m.intervals.qt_ms))
def detect(c, k, polmode, band_hi=15.0, wmin_ms=20):
    fs, rp = c["fs"], c["rp"]; qrs = (c["qrs"] or 90) / 1000; qt = (c["qt"] or 400) / 1000; t_end = qt - qrs / 2 + 0.02
    bb, aa = butter(2, [1.0 / (fs / 2), band_hi / (fs / 2)], btype="band"); y = filtfilt(bb, aa, c["x"])
    wins = [(rp[i] + int(t_end * fs), rp[i + 1] - int((qrs / 2 + 0.02) * fs)) for i in range(len(rp) - 1)]
    wins = [(a, b) for a, b in wins if b - a >= int(0.12 * fs)]
    if len(wins) < 2: return np.array([])
    dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
    noise = 1.4826 * float(np.median(np.abs(dia - np.median(dia)))) + 1e-9
    if polmode == "auto":
        signs = [np.sign((y[a:b] - np.median(y[a:b]))[np.argmax(np.abs(y[a:b] - np.median(y[a:b])))]) for a, b in wins]
        pols = [1.0 if np.sum(signs) >= 0 else -1.0]
    else:
        pols = [1.0, -1.0]
    best = []
    for pol in pols:
        ps = []
        h = max(k * noise, 15.0)
        for a, b in wins:
            seg = pol * (y[a:b] - np.median(y[a:b]))
            pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(wmin_ms / 1000 * fs), int(0.16 * fs)), distance=int(0.2 * fs))
            if len(pk): ps.append(a + pk[np.argmax(seg[pk])])        # one P per diastole (the largest)
        if len(ps) > len(best): best = ps
    return np.array(best)
def score(cs, **kw):
    g = tp = fp = 0; af_fp = 0; n_af = 0
    for c in cs:
        d = detect(c, **kw); d = d[(d >= c["lo"]) & (d <= c["hi"])] if len(d) else d
        tol = int(0.05 * c["fs"])
        if c["rhythm"].startswith("Atrial"):
            af_fp += len(d); n_af += 1; continue
        g += len(c["gold"]); tp += sum(np.any(np.abs(d - x) <= tol) for x in c["gold"]) if len(d) else 0
        fp += sum(not np.any(np.abs(c["gold"] - x) <= tol) for x in d) if len(c["gold"]) else len(d)
    return tp / max(1, g), tp / max(1, tp + fp), af_fp / max(1, n_af)
for par in (0, 1):
    cs = [c for c in cache if c["id"] % 2 == par]
    print("parity", par, "records", len(cs))
    for k, pm, bh in itertools.product((1.5, 2.0, 3.0, 4.0), ("auto", "both"), (15.0, 25.0)):
        r, p, a = score(cs, k=k, polmode=pm, band_hi=bh)
        print(f"  k={k} pol={pm} band<{bh:g}: recall {r:.2f} precision {p:.2f} | AF false P per record {a:.1f}")
