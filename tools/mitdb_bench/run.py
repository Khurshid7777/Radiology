"""D225: what the pipeline says on MIT-BIH Arrhythmia rhythm episodes (beat- and rhythm-annotated by
cardiologists) — junctional (NOD), idioventricular (IVR), second-degree AV block (BII) and sinus (N).

    python tools/mitdb_bench/run.py [--data data/mitdb] [--max 12] [--out reports/mitdb_bench/x.json]

10-s windows inside each episode, MLII as II plus the second lead (V1/V5), resampled 360 -> 500 Hz.
Download (open, ODC-By): the .hea/.dat/.atr files of https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/
into data/mitdb (git-ignored). A diagnostic, not a tuning set: these rhythms come from 1-3 patients each.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RHYTHMS = ("(NOD", "(IVR", "(BII", "(N")
RECORDS = ("124", "201", "207", "222", "231")


def episodes(path: str):
    import wfdb

    a = wfdb.rdann(path, "atr")
    out, cur, t0 = [], None, 0
    for s, sym, aux in zip(a.sample, a.symbol, a.aux_note, strict=True):
        if sym == "+":
            if cur:
                out.append((cur, t0, s))
            cur, t0 = aux.strip("\x00"), s
    out.append((cur, t0, 650000))
    return out


def window(path: str, s0: int):
    import wfdb
    from scipy.signal import resample_poly

    from cardiolens.contracts.signal import SignalRepresentation

    rec = wfdb.rdrecord(path, sampfrom=s0, sampto=s0 + 3600)
    x = resample_poly(rec.p_signal.T * 1000.0, 25, 18, axis=1)       # mV -> uV, 360 -> 500 Hz
    names = ["II" if n == "MLII" else n for n in rec.sig_name]
    return SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=500.0, lead_names=names,
                                duration_s=10.0, source="mitdb")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/mitdb")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from cardiolens.pipeline import run_pipeline

    res: dict[str, dict] = {}
    for r in RECORDS:
        path = str(a.data / r)
        n: collections.Counter = collections.Counter()
        for lab, s, e in episodes(path):
            if lab not in RHYTHMS:
                continue
            step = 3600 * (3 if lab == "(N" else 1)
            for s0 in range(s, e - 3600, step):
                if n[lab] >= a.max:
                    break
                n[lab] += 1
                key = f"{r} {lab}"
                d = res.setdefault(key, {"windows": 0, "codes": collections.Counter()})
                d["windows"] += 1
                try:
                    d["codes"].update(run_pipeline(window(path, s0), mode="A").findings.codes())
                except Exception as ex:  # a failure is a result
                    d["codes"][f"error:{type(ex).__name__}"] += 1
    out = {k: {"windows": v["windows"], "codes": dict(v["codes"].most_common(8))} for k, v in sorted(res.items())}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
