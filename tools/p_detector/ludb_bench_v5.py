"""D249 v5 on LUDB: v4 diastolic P + non-locked P on the template-subtracted residual (incl. P on T)."""
import sys, csv, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

from cardiolens.layer1_ingestion import load_wfdb
from cardiolens.layer3_measure.atrial import template_subtract
from cardiolens.pipeline import run_pipeline

rows = {r["ID"]: r for r in csv.DictReader(open("data/ludb/ludb.csv"))}
cache = []
for rid in sorted(rows, key=int):
    ann = wfdb.rdann(f"data/ludb/data/{rid}", "ii"); sym, smp = np.array(ann.symbol), np.array(ann.sample)
    sig = load_wfdb(f"data/ludb/data/{rid}"); res = run_pipeline(sig, mode="A"); m = res.measurements; s = res.signal
    cache.append(dict(id=int(rid), rhythm=rows[rid]["Rhythms"].strip().split("\n")[0], cond=rows[rid]["Conduction abnormalities"],
                      gold=smp[sym == "p"], lo=smp.min() - 50, hi=smp.max() + 50, x=np.asarray(s.lead("II"), float),
                      rp=np.asarray(m.r_peaks_samples, int), fs=s.sampling_rate_hz, qrs=m.intervals.qrs_ms, qt=m.intervals.qt_ms))
def v4(c, y, fs):
    rp = c["rp"]; qrs = (c["qrs"] or 90) / 1000; qt = (c["qt"] or 400) / 1000; t_end = qt - qrs / 2 + 0.02
    wins = [(rp[i] + int(t_end * fs), rp[i + 1] - int((qrs / 2 + 0.02) * fs)) for i in range(len(rp) - 1)]
    wins = [(a, b) for a, b in wins if b - a >= int(0.12 * fs)]
    if len(wins) < 2: return np.array([]), 0.0, 1.0
    dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
    noise = 1.4826 * float(np.median(np.abs(dia - np.median(dia)))) + 1e-9; h = max(2.0 * noise, 15.0)
    best, bc, bp = [], -1, 1.0
    for pl in (1.0, -1.0):
        cand = []
        for a, b in wins:
            seg = pl * (y[a:b] - np.median(y[a:b]))
            pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(0.02 * fs), int(0.16 * fs)), distance=int(0.2 * fs))
            if len(pk): cand.append(a + pk[np.argmax(seg[pk])])
        if len(cand) > bc: best, bc, bp = cand, len(cand), pl
    return np.array(best), noise, bp
def v5(c, k_res, use_res=True):
    fs, rp = c["fs"], c["rp"]
    bb, aa = butter(2, [1.0 / (fs / 2), 25.0 / (fs / 2)], btype="band"); y = filtfilt(bb, aa, c["x"])
    d4, noise, pol = v4(c, y, fs)
    if not use_res or len(rp) < 4: return d4
    res = template_subtract(c["x"], rp, fs, 0.30, 0.55)
    if res is None: return d4
    res = filtfilt(bb, aa, res)
    qrs = (c["qrs"] or 90) / 2000
    keep = np.ones(len(res), bool)
    for r in rp: keep[max(0, r - int((qrs + 0.03) * fs)):r + int((qrs + 0.07) * fs)] = False
    sd = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-9
    # P amplitude reference: v4 P heights in y
    amp = float(np.median([abs(y[p] - np.median(y[max(0, p - 50):p + 50])) for p in d4])) if len(d4) else 0.0
    h = max(k_res * sd, 0.5 * amp, 15.0)
    yy = pol * np.where(keep, res, 0.0)
    pk, _ = find_peaks(yy, height=h, prominence=h, width=(int(0.02 * fs), int(0.16 * fs)), distance=int(0.25 * fs))
    extra = [p for p in pk if not len(d4) or np.min(np.abs(d4 - p)) > int(0.12 * fs)]
    return np.array(sorted(list(d4) + extra))
def score(cs, **kw):
    g = tp = fp = 0; af_fp = n_af = 0
    for c in cs:
        d = v5(c, **kw); d = d[(d >= c["lo"]) & (d <= c["hi"])] if len(d) else d; tol = int(0.05 * c["fs"])
        if c["rhythm"].startswith("Atrial"): af_fp += len(d); n_af += 1; continue
        g += len(c["gold"]); tp += sum(np.any(np.abs(d - x) <= tol) for x in c["gold"]) if len(d) else 0
        fp += sum(not np.any(np.abs(c["gold"] - x) <= tol) for x in d) if len(c["gold"]) else len(d)
    return tp / max(1, g), tp / max(1, tp + fp), af_fp / max(1, n_af), g
for par in (0, 1):
    cs = [c for c in cache if c["id"] % 2 == par]
    av3 = [c for c in cache if "Iii degree" in c["cond"]]
    print("parity", par)
    print("  v4 all", ["%.2f" % v for v in score(cs, k_res=3, use_res=False)[:3]])
    for k in (3.0, 4.0, 5.0, 6.0):
        print(f"  v5 k_res={k}: all", ["%.2f" % v for v in score(cs, k_res=k)[:3]])
av3 = [c for c in cache if "Iii degree" in c["cond"]]
print("3AVB records", [c["id"] for c in av3], "gold P", sum(len(c["gold"]) for c in av3), "QRS", sum(len(c["rp"]) for c in av3))
print("  v4 on 3AVB: recall/precision", ["%.2f" % v for v in score(av3, k_res=3, use_res=False)[:2]])
for k in (3.0, 4.0, 5.0, 6.0):
    print(f"  v5 k_res={k} on 3AVB:", ["%.2f" % v for v in score(av3, k_res=k)[:2]])
