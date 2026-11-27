"""SD-167 prototype (version 6), kept as the record of what was measured: NOT part of the service, not linted or tested.
Run: python reports/scanner/sd167/panel_align_proto.py <scratch dir with lq_std_out/ and reg_head/> k/n"""
import json, sys
from pathlib import Path
import numpy as np
from ecgscan.bench.cases import load_case
from ecgscan.bench.metrics import clinical_measurements, truth_scan_result
from ecgscan.contract.models import Trace
from ecgscan.pipeline import scan

def envelope(x, fs, win_s=0.04):
    x = np.nan_to_num(np.asarray(x, float) - np.nanmedian(x))
    d = np.abs(np.diff(x, prepend=x[0]))
    w = max(1, int(win_s * fs))
    e = np.convolve(d, np.ones(w) / w, mode="same")
    return (e - e.mean()) / (e.std() + 1e-9)

def panel_shift(panel, rhythm, fs, max_shift_s=0.25):
    """(shift_s, peak) : moving the panel by shift_s best matches the rhythm strip's envelope."""
    p = panel.array(); r = rhythm.array()
    ep, er = envelope(p, fs), envelope(r, fs)
    i0 = int(round((panel.start_s - rhythm.start_s) * fs)); m = int(max_shift_s * fs); n = len(ep)
    best = (None, -9)
    for k in range(-m, m + 1):
        a = i0 + k
        if a < 0 or a + n > len(er): continue
        c = float(np.dot(ep, er[a:a + n]) / n)
        if c > best[1]: best = (k / fs, c)
    return best

ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def grid_pos(leads):
    """(row, col) per lead: columns by start time, rows by the standard order within a column."""
    starts = sorted({round(t.start_s, 1) for t in leads.values()})
    out = {}
    for c, s0 in enumerate(starts):
        col = sorted([l for l, t in leads.items() if round(t.start_s, 1) == s0], key=ORDER.index)
        for r, l in enumerate(col):
            out[l] = (r, c)
    return out

def beat_period(strip, fs, lo=0.25, hi=2.0):
    """The strip's RR (s) from its envelope's autocorrelation; None if it does not repeat clearly."""
    e = envelope(strip.array(), fs); n = len(e)
    a = np.array([float(np.dot(e[:n - k], e[k:]) / (n - k)) for k in range(int(hi * fs) + 1)])
    k0 = int(lo * fs); k = k0 + int(np.argmax(a[k0:]))
    return k / fs if a[k] > 0.3 * a[0] else None

def align_geo(res, fs, min_peak=0.3, max_shift=0.25, max_mad=0.02, min_n=6, min_span=0.03, rr_share=0.45):
    strips = [t for t in res.rhythm if len(t.samples_mv) / fs >= 8.0]
    if not strips or len(res.leads) < min_n: return res, []
    ref = strips[0]; pos = grid_pos(res.leads); est = []
    rr = beat_period(ref, fs)
    if rr is None: return res, []
    max_shift = min(max_shift, rr_share * rr)     # a shift near half a beat is ambiguous: a neighbour beat matches as well
    for lead, t in res.leads.items():
        s, peak = panel_shift(t, ref, fs, max_shift)
        if s is not None and peak >= min_peak: est.append((pos[lead][0], pos[lead][1], s, peak))
    if len(est) < min_n: return res, []
    X = np.array([[r, c, 1.0] for r, c, _, _ in est]); y = np.array([e[2] for e in est]); w = np.array([e[3] for e in est])
    keep = np.ones(len(y), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(X[keep] * w[keep, None], y[keep] * w[keep], rcond=None)
        res_ = y - X @ coef; mad = float(np.median(np.abs(res_[keep])))
        keep = np.abs(res_) <= max(3 * mad, 0.015)
        if keep.sum() < min_n: return res, []
    if mad > max_mad: return res, []
    shift = {lead: float(coef @ [pos[lead][0], pos[lead][1], 1.0]) for lead in res.leads}
    # the sheet as a whole: a skew worth correcting moves its windows apart by at least min_span, else nothing is touched
    if max(shift.values()) - min(shift.values()) < min_span: return res, []
    # time may not go below 0: the whole record (the strip with it) moves by one constant instead of clipping a column
    c = max(0.0, -min(t.start_s + shift[k] for k, t in res.leads.items()))
    out = res.model_copy(deep=True); log = []
    out.leads = {k: Trace.from_array(t.array(), start_s=t.start_s + shift[k] + c, fs=fs, lead=t.lead, quality=t.quality)
                 for k, t in res.leads.items()}
    if c:
        out.rhythm = [Trace.from_array(t.array(), start_s=t.start_s + c, fs=fs, lead=t.lead, quality=t.quality,
                                       flags=list(t.flags)) for t in res.rhythm]
    log = [(k, round(v * 1000)) for k, v in shift.items() if v] + ([("all", round(c * 1000))] if c else [])
    return out, log

def align(res, fs, min_shift=0.02, min_peak=0.3, max_shift=0.25):
    strips = [t for t in res.rhythm if len(t.samples_mv) / fs >= 8.0]
    if not strips: return res, []
    ref = strips[0]; out = res.model_copy(deep=True); log = []
    for lead, t in res.leads.items():
        s, peak = panel_shift(t, ref, fs, max_shift)
        if s is None or peak < min_peak or abs(s) < min_shift: continue
        out.leads = {**out.leads, lead: Trace.from_array(t.array(), start_s=max(0.0, t.start_s + s), fs=fs, lead=t.lead,
                                                        quality=t.quality)}
        log.append((lead, round(s * 1000)))
    return out, log

SP = Path(sys.argv[1]); rows = [json.loads(l) for l in open(SP / "reg_head/cases.jsonl")]
k, n = (int(v) for v in sys.argv[2].split("/"))
for i, row in enumerate(rows):
    if i % n != k or row["status"] == "rejected": continue
    case = load_case(SP / "lq_std_out/cases" / row["slice"] / row["case_id"])
    res = scan(str(case.image_path)); fs = res.sampling_rate_hz
    ref = clinical_measurements(truth_scan_result(case))
    before = clinical_measurements(res)
    al, log = align_geo(res, fs)
    after = clinical_measurements(al)
    err = lambda m, key: None if m[key] is None or ref[key] is None else round(abs(m[key] - ref[key]), 1)
    print(json.dumps({"case": row["case_id"], "slice": row["slice"], "status": row["status"],
                      **{f"{key}_before": err(before, key) for key in ("rr_ms", "qrs_ms", "qt_ms")},
                      **{f"{key}_after": err(after, key) for key in ("rr_ms", "qrs_ms", "qt_ms")}, "shifts": log}), flush=True)
