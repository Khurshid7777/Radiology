"""D227: VT signs on ventricular templates (INCART V beats) against aberrancy templates (bundle branch block
in sinus rhythm: PTB-XL CLBBB / CRBBB, INCART R beats).

    python tools/vt_signs_bench/run.py [--incart data/incartdb] [--ptbxl data/ptbxl_st data/ptbxl_rhythm] --out <json>

Per record: the median of its labelled beats (12 leads, 0.35 s before to 0.65 s after the R), QRS bounds from
the core's global fiducials, then ``vt_signs``. Rows to --out (no signal); summary printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def template(X: np.ndarray, beats, fs: float):
    from cardiolens.layer3_measure.global_beat import global_fiducials

    a, b = int(0.35 * fs), int(0.65 * fs)
    bs = [int(r) for r in beats if r - a >= 0 and r + b < X.shape[1]]
    if len(bs) < 5:
        return None
    T = np.median(np.stack([X[:, r - a:r + b] for r in bs]), axis=0)
    f = global_fiducials(T, a, fs, None)
    return T, a, f["qrs_on"], f["qrs_off"]


def incart_one(args):
    path, sym = args
    warnings.filterwarnings("ignore")
    import wfdb

    from vt_signs import vt_signs

    ann = wfdb.rdann(path, "atr")
    beats = [s for s, y in zip(ann.sample, ann.symbol, strict=True) if y == sym][:300]
    if len(beats) < 20:
        return None
    rec = wfdb.rdrecord(path)
    X = rec.p_signal.T * 1000.0
    names = [{"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(n, n) for n in rec.sig_name]
    X = X[[names.index(n) for n in NAMES]]
    t = template(X, beats, rec.fs)
    if t is None or t[2] is None or t[3] is None:
        return None
    T, _a, on, off = t
    return {"src": f"incart:{Path(path).name}:{sym}", "group": "ventricular" if sym == "V" else "aberrant",
            "idx": int(Path(path).name[1:]), "qrs_ms": round((off - on) / rec.fs * 1000.0),
            "signs": vt_signs(T, NAMES, int(on), int(off), rec.fs)}


def ptbxl_one(args):
    data, r = args
    warnings.filterwarnings("ignore")
    from vt_signs import vt_signs
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline

    sig = load_wfdb(str(Path(data) / r["filename_hr"]))
    res = run_pipeline(sig, mode="A")
    s = res.signal
    X = np.stack([s.lead(n).astype(float) for n in NAMES])
    t = template(X, res.measurements.r_peaks_samples or [], s.sampling_rate_hz)
    if t is None or t[2] is None or t[3] is None:
        return None
    T, _a, on, off = t
    return {"src": f"ptbxl:{r['ecg_id']}", "group": "aberrant", "idx": int(r["ecg_id"]),
            "labels": r["labels"], "qrs_ms": round((off - on) / s.sampling_rate_hz * 1000.0),
            "signs": vt_signs(T, NAMES, int(on), int(off), s.sampling_rate_hz)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incart", type=Path, default=ROOT / "data/incartdb")
    ap.add_argument("--ptbxl", type=Path, nargs="*", default=[ROOT / "data/ptbxl_st", ROOT / "data/ptbxl_rhythm"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs_i = [(str(p.with_suffix("")), sym) for p in sorted(a.incart.glob("*.atr")) for sym in ("V", "R")]
    jobs_p, seen = [], set()
    for d in a.ptbxl:
        for r in json.loads((d / "manifest.json").read_text())["records"]:
            L = set(r["labels"])
            if r["ecg_id"] in seen or not (L & {"CLBBB", "CRBBB"}) or not (L & {"SR", "SBRAD", "STACH"}) \
                    or L & {"PACE", "WPW", "AFIB", "AFLT", "PVC"}:
                continue
            seen.add(r["ecg_id"])
            jobs_p.append((str(d), r))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        rows += [x for x in ex.map(incart_one, jobs_i) if x]
        rows += [x for x in ex.map(ptbxl_one, jobs_p, chunksize=4) if x]
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(len(rows), "templates:", {g: sum(r["group"] == g for r in rows) for g in ("ventricular", "aberrant")})


if __name__ == "__main__":
    main()
