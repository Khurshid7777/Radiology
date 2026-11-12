"""D249 prototype v4: v3 + both P polarities, threshold 2 x diastolic SD, band 1-25 Hz, one P per diastole (tuned on LUDB even ids)."""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

LEADS = ("II", "III", "aVF", "I", "V1", "V2")
BAND = (1.0, 25.0)

def p_chain(leads, rpeaks, fs, qrs_ms=None, qt_ms=None, tol=0.12):
    rp = np.asarray(rpeaks, int)
    if len(rp) < 4:
        return None
    qrs = (qrs_ms or 90.0) / 1000.0
    qt = (qt_ms or 400.0) / 1000.0
    t_end = qt - qrs / 2 + 0.02                     # from R
    nyq = fs / 2.0
    bb, aa = butter(2, [BAND[0] / nyq, min(BAND[1] / nyq, 0.99)], btype="band")
    wmin, wmax = int(0.04 * fs), int(0.16 * fs)
    best = None
    for name in LEADS:
        x = leads.get(name)
        if x is None:
            continue
        y = filtfilt(bb, aa, np.asarray(x, float))
        wins = []
        for i in range(len(rp) - 1):
            a = rp[i] + int(t_end * fs); b = rp[i + 1] - int((qrs / 2 + 0.02) * fs)
            if b - a >= int(0.12 * fs):
                wins.append((a, b))
        if len(wins) < 3:
            continue
        dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
        noise = 1.4826 * float(np.median(np.abs(dia - np.median(dia)))) + 1e-9
        h = max(2.0 * noise, 15.0)
        ps, cover, pol = [], 0, 1.0
        for pl in (1.0, -1.0):
            cand, cv = [], 0
            for a, b in wins:
                seg = pl * (y[a:b] - np.median(y[a:b]))
                pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(0.02 * fs), wmax), distance=int(0.2 * fs))
                if len(pk):
                    cv += 1
                    cand.append(a + pk[np.argmax(seg[pk])])
            if cv > cover:
                ps, cover, pol = cand, cv, pl
        if len(ps) < 4:
            continue
        ps = np.array(sorted(ps)); pp = np.diff(ps) / fs; pp = pp[pp > 0.2]
        if len(pp) < 3:
            continue
        period = float(np.median(pp[pp <= 1.6 * np.percentile(pp, 25)]))
        nn = np.round(pp / period)
        regular = float(np.mean((nn >= 1) & (nn <= 3) & (np.abs(pp - nn * period) / period <= tol)))
        # P per QRS over the window span: P found / diastoles, scaled by how many atrial periods fit
        span = (rp[-1] - rp[0]) / fs
        ratio_rate = (60.0 / period) / (60.0 * (len(rp) - 1) / span)
        out = dict(lead=name, pol=pol, n_p=int(len(ps)), n_win=len(wins), cover=round(cover / len(wins), 3),
                   regular=round(regular, 3), rate_bpm=round(60.0 / period, 1), atrial_to_v=round(ratio_rate, 3),
                   amp_thr=round(h, 1), score=round(regular * cover / len(wins), 3))
        if best is None or out["score"] > best["score"]:
            best = out
    return best
