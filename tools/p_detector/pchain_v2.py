"""D249 prototype v2: full-cycle template; P-shaped waves inside it (locked P, incl. blocked 2:1 P) + residual P."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

LEADS = ("II", "III", "aVF", "V1")
BAND = (1.5, 12.0)

def p_chain(leads, rpeaks, fs, qrs_ms=None, tol=0.12, shape_min=0.75):
    rp = np.asarray(rpeaks, int)
    if len(rp) < 4:
        return None
    rr = np.diff(rp); med = float(np.median(rr))
    pre = int(0.35 * fs); post = int(med) - pre            # one full cycle from R-0.35 s
    if post < int(0.3 * fs):
        return None
    nyq = fs / 2.0
    bb, aa = butter(2, [BAND[0] / nyq, min(BAND[1] / nyq, 0.99)], btype="band")
    half = (qrs_ms or 0.0) / 2000.0
    w1 = int(max(0.06, half + 0.03) * fs); w2 = int(max(0.10, half + 0.07) * fs)
    hw = int(0.06 * fs)
    best = None
    for name in LEADS:
        x = leads.get(name)
        if x is None:
            continue
        y = filtfilt(bb, aa, np.asarray(x, float))
        segs = [y[r - pre:r + post] for r in rp if r - pre >= 0 and r + post <= len(y)]
        if len(segs) < 3:
            continue
        t = np.median(np.stack(segs), axis=0)
        res = y.copy()
        for r in rp:
            if r - pre >= 0 and r + post <= len(y):
                res[r - pre:r + post] -= t
        keep = np.ones(len(res), bool)
        for r in rp:
            keep[max(0, r - w1):r + w2] = False
        if keep.sum() < fs:
            continue
        sd = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-9
        # conducted P in the template: largest |deflection| 300-80 ms before R
        lo, hi = pre - int(0.30 * fs), pre - int(0.08 * fs)
        base = float(np.median(t[lo:hi]))
        k = lo + int(np.argmax(np.abs(t[lo:hi] - base)))
        amp = float(t[k] - base); has_tp = abs(amp) >= max(3 * sd, 25.0)
        locked = []
        if has_tp and k - hw >= 0:
            shape = t[k - hw:k + hw] - np.mean(t[k - hw:k + hw]); shape /= (np.linalg.norm(shape) + 1e-12)
            qrs_lo, qrs_hi = pre - w1, pre + w2
            for c in range(hw, len(t) - hw):
                if qrs_lo - hw <= c <= qrs_hi + hw:
                    continue
                s = t[c - hw:c + hw] - np.mean(t[c - hw:c + hw]); n = np.linalg.norm(s)
                if n < 1e-9:
                    continue
                corr = float(np.dot(s / n, shape)); a2 = float(t[c] - base)
                if corr >= shape_min and np.sign(a2) == np.sign(amp) and 0.5 <= abs(a2) / abs(amp) <= 1.6:
                    if not locked or c - locked[-1] > int(0.2 * fs):
                        locked.append(c)
                    elif abs(t[c] - base) > abs(t[locked[-1]] - base):
                        locked[-1] = c
        if has_tp and k not in locked:
            locked = sorted(set(locked) | {k})
        offs = [c - pre for c in locked]
        pols = [np.sign(amp)] if has_tp else [1.0, -1.0]
        for pol in pols:
            yy = pol * res; yy = np.where(keep, yy, 0.0)
            h = max(3 * sd, 0.5 * abs(amp)) if has_tp else 3 * sd
            pk, _ = find_peaks(yy, distance=int(0.25 * fs), height=h, prominence=h)
            lp = [r + o for r in rp for o in offs if 0 <= r + o < len(y)]
            nc = [p for p in pk if all(abs(p - c) > int(0.12 * fs) for c in lp)]
            ps = np.array(sorted(set(lp) | set(nc)))
            if len(ps) < 5:
                continue
            pp = np.diff(ps) / fs
            pp = pp[pp > 0.2]
            if len(pp) < 3:
                continue
            period = float(np.median(pp[pp <= 1.6 * np.percentile(pp, 25)]))
            nn = np.round(pp / period)
            regular = float(np.mean((nn >= 1) & (nn <= 3) & (np.abs(pp - nn * period) / period <= tol)))
            ratio = len(ps) / len(rp)
            out = dict(lead=name, has_tp=bool(has_tp), tp_amp=round(amp, 1), n_locked_per_beat=len(offs),
                       n_nc=len(nc), n_p=int(len(ps)), n_r=int(len(rp)), ratio=round(ratio, 3),
                       regular=round(regular, 3), rate_bpm=round(60.0 / period, 1), score=round(regular * min(1.0, ratio), 3))
            if best is None or out["score"] > best["score"]:
                best = out
    return best
