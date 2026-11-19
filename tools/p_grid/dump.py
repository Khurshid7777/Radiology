"""D313: per record — P-model probabilities (original lead and QRST-subtracted residual) for II and V1 at 100 Hz,
R peaks, QRS width, HR, codes. Numbers only; written to an .npz per shard for offline tuning.

    python tools/p_grid/dump.py {avb,ptb,slowvt} --out DIR [--shard k/n]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def items(which):
    if which == "avb":
        recs = json.loads((ROOT / "data/avb_bench/manifest.json").read_text())["records"]
        names = sorted(r["name"] for r in recs)
        idx = {n: i for i, n in enumerate(names)}
        return [(str(ROOT / "data/avb_bench" / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age"), idx[r["name"]] % 2)
                for r in recs]
    if which == "slowvt":
        recs = json.loads((ROOT / "data/slowvt_bench/manifest.json").read_text())["records"]
        return [(str(ROOT / "data/slowvt_bench" / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age"), 1) for r in recs]
    recs = json.loads((ROOT / "data/ptbxl_rhythm/manifest.json").read_text())["records"]
    keep = {"SBRAD", "NORM", "1AVB", "3AVB", "2AVB"}
    return [(str(ROOT / "data/ptbxl_rhythm" / r["filename_hr"]), r["ecg_id"], r["labels"], r.get("age"), int(r["ecg_id"]) % 2)
            for r in recs if set(r["labels"]) & keep]


def probs(x, fs):
    from cardiolens.layer3_measure.p_cnn import _prep, _session
    sess = _session()
    y = _prep(x, fs)
    logit = sess.run(None, {"x": y[None, None, :]})[0][0, 0]
    p = 1.0 / (1.0 + np.exp(-logit))
    return p[::5].astype(np.float16)            # 500 Hz -> 100 Hz


def envelope(res, fs):
    """|residual| in the P band (1.5-12 Hz), in robust SDs, at 100 Hz: a shape-free 'something is here' trace."""
    from scipy.signal import butter, filtfilt, resample_poly
    b, a = butter(2, [1.5 / (fs / 2), 12.0 / (fs / 2)], btype="band")
    y = np.abs(filtfilt(b, a, res))
    y = resample_poly(y, 100, int(round(fs)))
    s = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9
    return (y / s).astype(np.float16)


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer3_measure.atrial import template_subtract
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["avb", "ptb", "slowvt"])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    rows, arrays = [], {}
    for path, ident, labels, age, par in items(a.which)[k::n]:
        row = {"id": str(ident), "labels": labels, "odd": par}
        try:
            sig = load_wfdb(path)
            try:
                sig.patient.age_years = float(age) if age not in (None, "") else None
            except (TypeError, ValueError):
                pass
            res = run_pipeline(sig, mode="A"); m, s = res.measurements, res.signal
            fs = s.sampling_rate_hz
            rp = np.asarray(m.r_peaks_samples or [], int)
            row.update(codes=sorted(res.findings.codes()), alerts=sorted({x.code for x in res.safety.alerts}),
                       hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms, fs=fs, r100=[int(round(r * 100 / fs)) for r in rp])
            envs = []
            for ln in ("II", "III", "aVF", "V1", "V2"):
                if ln in s.lead_names and rp.size >= 4:
                    rr_ = template_subtract(s.lead(ln).astype(np.float64), rp, fs, 0.30, 0.55)
                    if rr_ is not None:
                        envs.append(envelope(rr_, fs).astype(float))
            if envs:
                L = min(len(e) for e in envs)
                arrays[f"{ident}|S|e"] = (sum(e[:L] for e in envs) / len(envs)).astype(np.float16)
            for ln in ("II", "V1"):
                if ln in s.lead_names and rp.size >= 4:
                    x = s.lead(ln).astype(np.float64)
                    arrays[f"{ident}|{ln}|o"] = probs(x, fs)
                    r = template_subtract(x, rp, fs, 0.30, 0.55)
                    if r is not None:
                        arrays[f"{ident}|{ln}|r"] = probs(r, fs)
                        arrays[f"{ident}|{ln}|e"] = envelope(r, fs)
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / f"{a.which}_{k}.npz", **arrays)
    (a.out / f"{a.which}_{k}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
