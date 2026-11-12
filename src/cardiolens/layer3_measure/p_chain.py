"""P waves in diastole, per lead (D249 v4; D252 uses it as the sinus-mechanism check).

In each lead (II, III, aVF, I, V1, V2; band 1-25 Hz) the diastole of every cycle (estimated T end ..
next QRS onset) is searched for one P-like wave of either polarity, above 2 x the diastolic robust SD.
The best lead (regular P-P x share of cycles with a P) is reported. Checked against the LUDB beat
annotations (lead II, held-out odd ids): recall 0.86, precision 0.95 (tools/p_detector/ludb_bench.py).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

P_CHAIN_LEADS = ("II", "III", "aVF", "I", "V1", "V2")
P_CHAIN_BAND_HZ = (1.0, 25.0)
P_CHAIN_SD_K = 2.0
P_CHAIN_MIN_UV = 15.0
P_CHAIN_PERIOD_TOL = 0.12


def p_chain(leads: dict[str, np.ndarray], rpeaks, fs: float, qrs_ms: float | None = None,
            qt_ms: float | None = None) -> dict | None:
    """Best-lead P chain: ``cover`` (share of diastoles with a P), ``regular`` (P-P regularity), ``score``
    (their product), ``rate_bpm``; None without at least 3 searchable diastoles in some lead."""
    rp = np.asarray(rpeaks, int)
    if len(rp) < 4:
        return None
    qrs = (qrs_ms or 90.0) / 1000.0
    qt = (qt_ms or 400.0) / 1000.0
    t_end = qt - qrs / 2 + 0.02
    nyq = fs / 2.0
    bb, aa = butter(2, [P_CHAIN_BAND_HZ[0] / nyq, min(P_CHAIN_BAND_HZ[1] / nyq, 0.99)], btype="band")
    wmax = int(0.16 * fs)
    best = None
    for name in P_CHAIN_LEADS:
        x = leads.get(name)
        if x is None or len(x) < 3 * wmax:
            continue
        y = filtfilt(bb, aa, np.asarray(x, float))
        wins = [(int(rp[i] + t_end * fs), int(rp[i + 1] - (qrs / 2 + 0.02) * fs)) for i in range(len(rp) - 1)]
        wins = [(a, b) for a, b in wins if b - a >= int(0.12 * fs) and a >= 0 and b <= len(y)]
        if len(wins) < 3:
            continue
        dia = np.concatenate([y[a:b] - np.median(y[a:b]) for a, b in wins])
        h = max(P_CHAIN_SD_K * 1.4826 * float(np.median(np.abs(dia - np.median(dia)))), P_CHAIN_MIN_UV)
        ps: list[int] = []
        cover = 0
        for pol in (1.0, -1.0):
            cand, cv = [], 0
            for a, b in wins:
                seg = pol * (y[a:b] - np.median(y[a:b]))
                pk, _ = find_peaks(seg, height=h, prominence=h, width=(int(0.02 * fs), wmax),
                                   distance=int(0.2 * fs))
                if len(pk):
                    cv += 1
                    cand.append(a + int(pk[np.argmax(seg[pk])]))
            if cv > cover:
                ps, cover = cand, cv
        out = {"lead": name, "cover": round(cover / len(wins), 3), "regular": 0.0, "rate_bpm": None,
               "score": 0.0, "n_windows": len(wins)}
        pp = np.diff(np.array(sorted(ps))) / fs if len(ps) >= 2 else np.array([])
        pp = pp[pp > 0.2]
        if len(pp) >= 3:
            period = float(np.median(pp[pp <= 1.6 * np.percentile(pp, 25)]))
            nn = np.round(pp / period)
            regular = float(np.mean((nn >= 1) & (nn <= 3) & (np.abs(pp - nn * period) / period <= P_CHAIN_PERIOD_TOL)))
            out.update(regular=round(regular, 3), rate_bpm=round(60.0 / period, 1),
                       score=round(regular * cover / len(wins), 3))
        if best is None or (out["score"], out["cover"]) > (best["score"], best["cover"]):
            best = out
    return best
