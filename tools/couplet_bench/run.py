"""D253: ventricular couplets on 10-s windows of MIT-BIH and INCART (cardiologists' beat labels).

    python tools/couplet_bench/run.py --out <json> [--per-record 6]

Truth per window: V-V pairs not inside a run of >= 3 V. Windows: "couplet" (>= 1 pair, no run >= 3) and
"pvc_only" (>= 1 V, no pair, no run). The pipeline runs on each window; the rows keep the core's PVC beat
indices (``detect_pvc_beats``) so the couplet rule is scored offline. Data in data/mitdb, data/incartdb
(git-ignored, PhysioNet ODC-By).
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
INCART_NAMES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def truth(sym: list[str]) -> tuple[int, int, int]:
    """(pairs, runs >= 3, PVCs) of a beat-symbol list."""
    pairs = runs = run = 0
    for s in [*sym, "N"]:
        if s == "V":
            run += 1
            continue
        if run == 2:
            pairs += 1
        elif run >= 3:
            runs += 1
        run = 0
    return pairs, runs, sym.count("V")


def load(path: str, s0: int, n: int):
    import wfdb
    from scipy.signal import resample_poly

    from cardiolens.contracts.signal import SignalRepresentation

    rec = wfdb.rdrecord(path, sampfrom=s0, sampto=s0 + n)
    fs = rec.fs
    x = resample_poly(rec.p_signal.T * 1000.0, 500, int(fs), axis=1)
    names = [INCART_NAMES.get(nm, "II" if nm == "MLII" else nm) for nm in rec.sig_name]
    return SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=500.0, lead_names=names,
                                duration_s=n / fs, source="bench")


def main() -> None:
    warnings.filterwarnings("ignore")
    import wfdb

    from cardiolens.arrhythmia.beats import detect_pvc_beats, detect_ventricular_couplets
    from cardiolens.pipeline import run_pipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-record", type=int, default=6)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, nsh = (int(v) for v in a.shard.split("/"))
    paths = sorted(str(p)[:-4] for d in ("data/mitdb", "data/incartdb") for p in (ROOT / d).glob("*.hea"))
    rows = []
    for path in paths[k::nsh]:
        try:
            ann = wfdb.rdann(path, "atr")
            fs = wfdb.rdheader(path).fs
        except Exception as ex:  # a record without annotations is skipped, and said so
            print(f"skip {path}: {ex}", file=sys.stderr)
            continue
        beats = [(s, y) for s, y in zip(ann.sample, ann.symbol, strict=True) if y in "NLRVFEjJAaSe/fQ"]
        n = int(10 * fs)
        cnt = {"couplet": 0, "pvc_only": 0}
        last = int(beats[-1][0]) if beats else 0
        for s0 in range(int(fs), last - n, n):
            sym = [y for s, y in beats if s0 <= s < s0 + n]
            pairs, runs, pvcs = truth(sym)
            kind = "couplet" if pairs and not runs else ("pvc_only" if pvcs and not pairs and not runs else None)
            if kind is None or cnt[kind] >= a.per_record:
                continue
            cnt[kind] += 1
            try:
                res = run_pipeline(load(path, s0, n), mode="A")
                m = res.measurements
                pvc = detect_pvc_beats(res.signal, m.r_peaks_samples or [], res.signal.sampling_rate_hz)
                rows.append({"rec": Path(path).name, "db": Path(path).parent.name, "s0": int(s0), "kind": kind,
                             "pairs": pairs, "pvcs": pvcs, "n_beats": len(m.r_peaks_samples or []),
                             "pvc_idx": [int(i) for i in pvc],
                             "couplets": [int(i) for i in detect_ventricular_couplets(
                                 res.signal, m.r_peaks_samples or [], res.signal.sampling_rate_hz)],
                             "grid": {f"{c}/{pr}/{po}": len(detect_ventricular_couplets(
                                 res.signal, m.r_peaks_samples or [], res.signal.sampling_rate_hz,
                                 corr_max=c, pre_max=pr, post_min=po))
                                 for c in (0.8, 0.85, 0.9) for pr in (1.0, 1.1) for po in (0.0, 0.9, 1.0)},
                             "codes": sorted(res.findings.codes()),
                             "alerts": [x.code for x in res.safety.alerts]})
            except Exception as ex:
                rows.append({"rec": Path(path).name, "kind": kind, "error": f"{type(ex).__name__}: {str(ex)[:80]}"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
