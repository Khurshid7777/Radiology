"""SD-170 prototype (last variant), the record of what was measured: NOT part of the service, not linted or tested.
Run: SM=2.5 python reports/scanner/sd170/dewarp_proto.py <scratch dir with lq_std_out/> <slice> ..."""
import json, sys
from pathlib import Path
import numpy as np
from ecgscan.bench.cases import load_case
from ecgscan.bench.metrics import compare_trace
from ecgscan.pipeline import ScanDebug, scan
from ecgscan.vectorize.trace import resample_nan

def box(x, w):
    w = max(1, int(w)); c = np.cumsum(np.concatenate([[0.0], np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")]))
    return (c[w:] - c[:-w]) / w

def grid_dx(bgr, y0, y1, x0, x1, ppm, period_mm=5.0, smooth_mm=10.0, pad_mm=8.0):
    """dx(x) for x0..x1 (px): where the paper moved, from the 5-mm grid lines' local phase (trend removed)."""
    H, W = bgr.shape[:2]
    pad = int(pad_mm * ppm); a, b = max(0, x0 - pad), min(W, x1 + pad)
    band = bgr[max(0, int(y0)):min(H, int(y1)), a:b].astype(float)
    dark = 255.0 - band[..., 1] if band.ndim == 3 else 255.0 - band         # red grid: low green
    p = np.median(dark, axis=0)
    p = p - box(p, 5 * ppm)
    P = period_mm * ppm; x = np.arange(len(p))
    z = p * np.exp(-2j * np.pi * x / P)
    zs = box(z.real, smooth_mm * ppm) + 1j * box(z.imag, smooth_mm * ppm)
    ph = np.unwrap(np.angle(zs))
    amp = np.abs(zs)
    k = np.polyfit(x, ph, 1, w=amp + 1e-9)
    res = ph - np.polyval(k, x)
    dx = -res * P / (2 * np.pi)
    return dx[x0 - a: x1 - a], float(np.median(amp) / (np.median(np.abs(p)) + 1e-9))

FIXED_SLOPE = False
TWO_D = False
ROW_WIDE = True

def grid_dx2(bgr, y0, y1, x0, x1, ppm, pad_mm=8.0, smooth_mm=float(__import__("os").environ.get("SM", "4")), max_step=0.35):
    """dx(x) for x0..x1 (px) from the 1-mm vertical lines themselves: each line found, numbered by its neighbour
    (the paper moves < 0.35 mm per mm), the lattice a + b*k fitted, the residuals = where the paper moved."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    H, W = bgr.shape[:2]
    pad = int(pad_mm * ppm); a, b = max(0, x0 - pad), min(W, x1 + pad)
    band = bgr[max(0, int(y0)):min(H, int(y1)), a:b].astype(float)
    dark = 255.0 - band[..., 1] if band.ndim == 3 else 255.0 - band
    p = np.median(dark, axis=0)
    p = gaussian_filter1d(p - box(p, 3 * ppm), 0.15 * ppm)
    pk, _ = find_peaks(p, distance=0.6 * ppm, prominence=max(1e-6, 0.2 * np.std(p)))
    if len(pk) < 10: return np.zeros(x1 - x0), 0.0
    # sub-pixel line centres
    pos = []
    for i in pk:
        if 0 < i < len(p) - 1:
            y_0, y_1, y_2 = p[i - 1], p[i], p[i + 1]; den = y_0 - 2 * y_1 + y_2
            pos.append(i + (0.5 * (y_0 - y_2) / den if den < 0 else 0.0))
    pos = np.array(pos)
    k = np.zeros(len(pos), int)
    for j in range(1, len(pos)):
        k[j] = k[j - 1] + max(1, int(round((pos[j] - pos[j - 1]) / ppm)))
    steps = np.diff(pos) / np.maximum(1, np.diff(k))
    good = np.concatenate([[True], np.abs(steps / ppm - 1) < max_step])
    pos, k = pos[good], k[good]
    if len(pos) < 10: return np.zeros(x1 - x0), 0.0
    if FIXED_SLOPE:
        res = pos - k * ppm                   # the sheet is scaled to exactly ppm px per mm: only the offset is free
        # a line missed in the count moves the rest by a whole millimetre: neighbours never differ by half of one
        d = np.diff(res); d = (d + ppm / 2) % ppm - ppm / 2
        res = res[0] + np.concatenate([[0.0], np.cumsum(d)])
    else:
        res = pos - k * ppm
        d = np.diff(res); d = (d + ppm / 2) % ppm - ppm / 2
        res = res[0] + np.concatenate([[0.0], np.cumsum(d)])
        res = res - np.polyval(np.polyfit(k, res, 1), k)     # only the uneven part: a straight drift is the scale's
    res = res - np.median(res)
    x = np.arange(len(p))
    dx = np.interp(x, pos, res)
    dx = gaussian_filter1d(dx, smooth_mm * ppm)
    return dx[x0 - a: x1 - a], float(len(pos) / max(1, k[-1] - k[0] + 1))

def retime(y, dx, x0, iso, ppm, gain, fs_col, fs_out, dur):
    mv = (iso - y) / (ppm * gain)
    i = np.arange(len(y), dtype=float)
    u = i - dx                                       # true position of each column
    ok = np.isfinite(mv)
    if ok.sum() < 10: return resample_nan(mv, fs_col, fs_out, dur)
    u = np.maximum.accumulate(u) + 1e-9 * i         # time only goes forward (a steep stroke may fold the field)
    mv2 = np.interp(i, u[ok], mv[ok], left=np.nan, right=np.nan)
    # keep gaps as gaps: a corrected column whose nearest original column was missing stays missing
    near = np.clip(np.round(np.interp(i, u, i)).astype(int), 0, len(y) - 1)
    mv2[~ok[near]] = np.nan
    return resample_nan(mv2, fs_col, fs_out, dur)

SP = Path(sys.argv[1]); fs_out = 500.0
for sl in sys.argv[2:]:
    for cdir in sorted((SP / "lq_std_out/cases" / sl).iterdir()):
        case = load_case(cdir); dbg = ScanDebug(); res = scan(str(case.image_path), debug=dbg)
        if res.status == "rejected" or not dbg.readings: continue
        rect = dbg.rectified; ppm = rect.px_per_mm
        speed = float(res.calibration.paper_speed_mm_s or 25); gain = float(res.calibration.gain_mm_mv or 10)
        fs_col = ppm * speed
        before, after, dxs = [], [], []
        rowdx = {}
        truth = {(t.lead, t.kind): t for t in case.truth.traces}
        for r in dbg.readings:
            tr = truth.get((r.panel.lead, r.panel.kind))
            if tr is None: continue
            p = r.panel; band = 8.0 * ppm
            if TWO_D:
                xs0 = min(int(q.panel.x0) for q in dbg.readings); xs1 = max(int(q.panel.x0) + len(q.y_px) for q in dbg.readings)
                key = round(p.baseline)
                if key not in rowdx:
                    ys = p.baseline + np.arange(-4, 5) * 4.0 * ppm
                    rowdx[key] = (np.array([grid_dx2(rect.bgr, yc - 3 * ppm, yc + 3 * ppm, xs0, xs1, ppm)[0] for yc in ys]), xs0, ys)
                field, xs0, ys = rowdx[key]
                sl = field[:, int(p.x0) - xs0: int(p.x0) - xs0 + len(r.y_px)]
                yy = np.where(np.isfinite(r.y_px), r.y_px, p.baseline)
                fi = np.clip((yy - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.000001)
                i0 = np.floor(fi).astype(int); f = fi - i0
                cols = np.arange(sl.shape[1])
                dx = sl[i0, cols] * (1 - f) + sl[np.minimum(i0 + 1, len(ys) - 1), cols] * f
            elif ROW_WIDE:
                xs0 = min(int(q.panel.x0) for q in dbg.readings); xs1 = max(int(q.panel.x0) + len(q.y_px) for q in dbg.readings)
                key = round(p.baseline)
                if key not in rowdx:
                    rowdx[key] = (grid_dx2(rect.bgr, p.baseline - band, p.baseline + band, xs0, xs1, ppm)[0], xs0)
                full, xs0 = rowdx[key]
                dx = full[int(p.x0) - xs0: int(p.x0) - xs0 + len(r.y_px)]
            else:
                dx, q = grid_dx2(rect.bgr, p.baseline - band, p.baseline + band, int(p.x0), int(p.x0) + len(r.y_px), ppm)
            y = r.y_px; iso = float(np.nanmedian(y)) if np.isfinite(y).any() else p.baseline
            dur = p.t1_s - p.t0_s
            s0 = resample_nan((iso - y) / (ppm * gain), fs_col, fs_out, dur)
            big = float(np.percentile(np.abs(dx), 90)) / ppm >= float(__import__("os").environ.get("MINDX", "0.25"))
            s1 = retime(y, dx, p.x0, iso, ppm, gain, fs_col, fs_out, dur) if big else s0
            for s, acc in ((s0, before), (s1, after)):
                c = compare_trace(np.asarray(tr.samples_mv, float), tr.start_s, s, p.t0_s, fs_out, max_lag_s=0.25, min_overlap=0.5, snr_cap_db=40)
                acc.append(None if c is None else c["r"])
            dxs.append(float(np.max(np.abs(dx))) / ppm)
        med = lambda v: round(float(np.median([x for x in v if x is not None])), 3)
        print(json.dumps({"case": cdir.name, "r_before": med(before), "r_after": med(after), "max_dx_mm_median": round(float(np.median(dxs)), 2)}), flush=True)
