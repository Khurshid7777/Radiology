import json, sys
from pathlib import Path
import numpy as np
from ecgscan.bench.cases import load_case
from ecgscan.bench.metrics import _pick_output
from ecgscan.pipeline import scan

SP = Path(sys.argv[1])
def windowed(truth, t0, out, o0, fs, lag_g, win=0.5, loc=0.08):
    """r per 0.5-s window of the truth with the read allowed its own shift there (global lag first)."""
    g0, oo = int(round(t0 * fs)), int(round(o0 * fs)); L = int(round(lag_g * fs)); w = int(win * fs); m = int(loc * fs)
    rs_fixed, rs_free, lags = [], [], []
    for a in range(0, len(truth) - w, w):
        g = truth[a:a + w]
        if g.std() < 1e-6: continue
        best, fixed = None, None
        for k in range(-m, m + 1):
            i = g0 + a - oo - L + k
            if i < 0 or i + w > len(out): continue
            d = out[i:i + w]; ok = np.isfinite(d)
            if ok.sum() < 0.8 * w or d[ok].std() < 1e-6: continue
            r = float(np.corrcoef(g[ok], d[ok])[0, 1])
            if k == 0: fixed = r
            if best is None or r > best[0]: best = (r, k)
        if best is None or fixed is None: continue
        rs_fixed.append(fixed); rs_free.append(best[0]); lags.append(best[1] / fs)
    return rs_fixed, rs_free, lags
for sl in sys.argv[2:]:
    allf, allr, alll = [], [], []
    for cdir in sorted((SP / "lq_std_out/cases" / sl).iterdir()):
        case = load_case(cdir); res = scan(str(case.image_path)); fs = case.truth.fs
        if res.status == "rejected": continue
        for tr in case.truth.traces:
            out = _pick_output(res, tr.lead, tr.kind)
            if out is None: continue
            from ecgscan.bench.metrics import compare_trace
            c = compare_trace(tr.samples_mv, tr.start_s, out.array(), out.start_s, fs, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=40)
            if c is None: continue
            f, r, l = windowed(np.asarray(tr.samples_mv, float), tr.start_s, out.array(), out.start_s, fs, c["lag_s"])
            allf += f; allr += r; alll += l
    q = lambda v, p: round(float(np.percentile(v, p)), 3)
    print(json.dumps({"slice": sl, "windows": len(allf), "r_fixed_median": q(allf, 50), "r_fixed_p10": q(allf, 10),
                      "r_free_median": q(allr, 50), "r_free_p10": q(allr, 10), "abs_local_lag_ms_median": round(1000 * float(np.median(np.abs(alll))), 1),
                      "abs_local_lag_ms_p90": round(1000 * float(np.percentile(np.abs(alll), 90)), 1)}), flush=True)
