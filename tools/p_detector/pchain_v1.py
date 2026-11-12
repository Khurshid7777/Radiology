"""D249 prototype: the full P chain = conducted P (from the beat template) + non-conducted P (residual peaks)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from cardiolens.layer3_measure.atrial import template_subtract

LEADS = ("II", "III", "aVF", "V1")
BAND = (1.5, 12.0)
WIN = (0.30, 0.55)

def _template(y, rp, fs):
    a, b = int(WIN[0] * fs), int(WIN[1] * fs)
    segs = [y[r - a:r + b] for r in rp if r - a >= 0 and r + b <= len(y)]
    return (np.median(np.stack(segs), axis=0), a) if len(segs) >= 3 else (None, None)

def p_chain(leads, rpeaks, fs, qrs_ms=None, tol=0.12):
    rp = np.asarray(rpeaks, int)
    if len(rp) < 4:
        return None
    nyq = fs / 2.0
    bb, aa = butter(2, [BAND[0] / nyq, min(BAND[1] / nyq, 0.99)], btype="band")
    half = (qrs_ms or 0.0) / 2000.0
    w1 = int(max(0.06, half + 0.03) * fs); w2 = int(max(0.10, half + 0.07) * fs)
    best = None
    for name in LEADS:
        x = leads.get(name)
        if x is None:
            continue
        y = filtfilt(bb, aa, np.asarray(x, float))
        t, a = _template(y, rp, fs)
        if t is None:
            continue
        res = template_subtract(x, rp, fs, *WIN)
        if res is None:
            continue
        res = filtfilt(bb, aa, res)
        keep = np.ones(len(res), bool)
        for r in rp:
            keep[max(0, r - w1):r + w2] = False
        if keep.sum() < fs:
            continue
        sd = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-9
        # template P: largest |deflection| 300-80 ms before R, baseline = template median there
        lo, hi = a - int(0.30 * fs), a - int(0.08 * fs)
        seg = t[lo:hi] - np.median(t[max(0, lo - int(0.1 * fs)):lo + 1])
        k = int(np.argmax(np.abs(seg)))
        amp = float(seg[k]); has_tp = abs(amp) >= max(3 * sd, 25.0)
        d = (lo + k) - a   # samples relative to R (negative)
        pols = [np.sign(amp)] if has_tp else [1.0, -1.0]
        for pol in pols:
            yy = pol * res.copy(); yy[~keep] = 0.0
            h = max(3 * sd, 0.5 * abs(amp)) if has_tp else 3 * sd
            pk, _ = find_peaks(yy, distance=int(0.25 * fs), height=h, prominence=h)
            cond = [r + d for r in rp] if has_tp else []
            nc = [p for p in pk if all(abs(p - c) > int(0.12 * fs) for c in cond)]
            ps = np.array(sorted(list(cond) + list(nc)))
            if len(ps) < 5:
                continue
            pp = np.diff(ps) / fs
            period = float(np.median(pp[pp <= 1.6 * np.percentile(pp, 25)]))
            if period <= 0:
                continue
            nn = np.round(pp / period)
            regular = float(np.mean((nn >= 1) & (nn <= 3) & (np.abs(pp - nn * period) / period <= tol)))
            ratio = len(ps) / len(rp)
            out = dict(lead=name, has_tp=bool(has_tp), tp_amp=round(amp, 1), n_cond=len(cond), n_nc=len(nc),
                       n_p=int(len(ps)), n_r=int(len(rp)), ratio=round(ratio, 3), regular=round(regular, 3),
                       rate_bpm=round(60.0 / period, 1), score=round(regular * min(1.0, ratio), 3))
            if best is None or out["score"] > best["score"]:
                best = out
    return best
