"""D270 study: P-wave morphology per record — what tells a sinus P from an ectopic (atrial tachycardia) or an
absent / retrograde one (AV-nodal SVT).

    python tools/p_morph/study.py chapman --out <rows.jsonl> [--shard k/n]
    python tools/p_morph/study.py ptbxl --out <rows.jsonl> [--shard k/n]

From the median beat (all leads): the isoelectric level in the PR segment, then in the P window (QRS onset
-300 ms .. -40 ms, not before 45 % of the RR) the signed largest deflection per lead, its size against the
diastolic noise, the P frontal axis from the limb leads' signed areas, the post-QRS deflection (ST start,
retrograde P), plus the core's own measurements (PR, P onset share, P axis, p_chain). Rows carry no signal.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2")
LIMB_DEG = {"I": 0.0, "II": 60.0, "III": 120.0, "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}


def morphology(sig, m) -> dict:
    from cardiolens.layer3_measure.global_beat import global_fiducials, median_beats

    fs = sig.sampling_rate_hz
    rp = np.asarray(m.r_peaks_samples or [], int)
    out: dict = {}
    if rp.size < 4:
        return out
    X, a = median_beats(sig.signal, rp, fs)
    if X is None:
        return out
    rr_s = float(np.median(np.diff(rp))) / fs
    fid = global_fiducials(X, a, fs, rr_s)
    on, off = fid["qrs_on"], fid["qrs_off"]
    p0 = max(0, on - int(0.30 * fs), a - int(0.45 * rr_s * fs))
    p1 = on - int(0.04 * fs)
    if p1 - p0 < int(0.04 * fs):
        return out
    names = list(sig.lead_names)
    area = {}
    for ln in LEADS:
        if ln not in names:
            continue
        x = X[names.index(ln)].astype(float)
        raw = x[p0:p1]
        # Detrended against the straight line across the window: a T-wave tail or wander sloping through
        # the window is not taken for P (as the core's P axis, D99).
        seg = raw - np.linspace(raw[0], raw[-1], raw.size)
        base = float(np.median(x[max(0, on - int(0.03 * fs)):on]))      # PR segment end, for the post-QRS look
        noise = float(np.std(np.diff(x[p0:p1]))) + 1e-6
        k = int(np.argmax(np.abs(seg)))
        out[f"p_{ln}"] = round(float(seg[k]), 1)                         # signed peak, uV
        out[f"snr_{ln}"] = round(float(abs(seg[k]) / (noise * 5)), 2)
        area[ln] = float(np.mean(seg))
        if off is not None and off + int(0.12 * fs) < x.size:
            post = x[off:off + int(0.12 * fs)] - base
            j = int(np.argmax(np.abs(post)))
            out[f"post_{ln}"] = round(float(post[j]), 1)
    xs = sum(area[ln] * np.cos(np.radians(d)) for ln, d in LIMB_DEG.items() if ln in area)
    ys = sum(area[ln] * np.sin(np.radians(d)) for ln, d in LIMB_DEG.items() if ln in area)
    if abs(xs) + abs(ys) > 1e-6:
        out["p_axis_med"] = round(float(np.degrees(np.arctan2(ys, xs))), 1)
    out["p_win_ms"] = round((p1 - p0) / fs * 1000.0)
    return out


def row_for(sig, labels, ident) -> dict:
    from cardiolens.layer3_measure.p_chain import P_CHAIN_LEADS, p_chain
    from cardiolens.pipeline import run_pipeline

    res = run_pipeline(sig, mode="A")
    m, s = res.measurements, res.signal
    rp = np.asarray(m.r_peaks_samples or [])
    rr = np.diff(rp)
    pc = p_chain({ln: s.lead(ln) for ln in P_CHAIN_LEADS if ln in s.lead_names}, m.r_peaks_samples or [],
                 s.sampling_rate_hz, m.intervals.qrs_ms, m.intervals.qt_ms)
    row = {"id": ident, "labels": labels, "codes": sorted(res.findings.codes()), "hr": m.heart_rate_bpm,
           "qrs": m.intervals.qrs_ms, "pr": m.intervals.pr_ms, "pon": m.p_onset_ratio, "prstd": m.pr_std_ms,
           "p_axis": getattr(m.axes, "p_deg", None) if getattr(m, "axes", None) else None,
           "cv": float(rr.std() / rr.mean()) if rr.size > 2 else None,
           "pc_cover": pc["cover"] if pc else None, "pc_regular": pc["regular"] if pc else None}
    row.update(morphology(s, m))
    return row


def main() -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb

    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["chapman", "ptbxl"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    if a.which == "chapman":
        recs = json.loads((ROOT / "data/svt_bench/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/svt_bench" / r["folder"] / r["name"]), r["labels"], r["name"], r.get("age")) for r in recs]
    else:
        keep = {"SR", "STACH", "SVTAC", "PSVT", "AFLT", "AFIB", "SARRH", "PAC", "SBRAD", "NORM"}
        recs = json.loads((ROOT / "data/ptbxl_rhythm/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/ptbxl_rhythm" / r["filename_hr"]), r["labels"], r["ecg_id"], r.get("age"))
                 for r in recs if set(r["labels"]) & keep]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for path, labels, ident, age in items[k::n]:
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                fh.write(json.dumps(row_for(sig, labels, ident)) + "\n")
            except Exception as ex:
                fh.write(json.dumps({"id": ident, "labels": labels, "error": f"{type(ex).__name__}: {str(ex)[:60]}"}) + "\n")


if __name__ == "__main__":
    main()
