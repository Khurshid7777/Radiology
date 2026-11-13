"""D262: ventricular tachycardia on 10-s windows — MIT-BIH, VFDB ("(VT" rhythm), INCART, SVDB (V-beat runs).

    python tools/vt_bench/run.py --out <rows.json> [--shard i/n]
    python tools/vt_bench/run.py score <rows.json> [<rows.json> ...]

Classes: vt_full (window entirely inside "(VT"), vt_run (a run of >= 3 V beats, not entirely in VT),
control (no V beat, no VT / VFL / VF rhythm). Data in data/mitdb, data/vf_bench/vfdb, data/incartdb, data/svdb
(git-ignored, PhysioNet ODC-By). Rows keep codes, alerts, HR and QRS only (no signal).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DBS = {"mitdb": "data/mitdb", "vfdb": "data/vf_bench/vfdb", "incartdb": "data/incartdb", "svdb": "data/svdb"}
NAMES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II", "MLIII": "III", "MLI": "I"}
VT_CODES = {"ventricular_tachycardia", "wide_complex_tachycardia", "wide_qrs_tachycardia", "ventricular_fibrillation"}
CAP = {"vt_full": 10, "vt_run": 6, "control": 3}
WIN_S = 10.0
BEATS = "NLRVFEjJAaSe/fQ"


def lead_names(names: list[str]) -> list[str]:
    out, spare = [], ["II", "V1", "V5"]
    for nm in names:
        nm = NAMES.get(nm, nm)
        if nm in ("ECG", "ECG1", "ECG2") or nm in out:
            nm = next(s for s in spare if s not in out)
        out.append(nm)
    return out


def load(path: str, s0: int, n: int):
    import wfdb
    from scipy.signal import resample_poly

    from cardiolens.contracts.signal import SignalRepresentation

    rec = wfdb.rdrecord(path, sampfrom=s0, sampto=s0 + n)
    x = resample_poly(rec.p_signal.T * 1000.0, 500, int(rec.fs), axis=1)
    return SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=500.0,
                                lead_names=lead_names(rec.sig_name), duration_s=n / rec.fs, source="bench")


def rhythm_spans(ann, n: int) -> list[tuple[int, int, str]]:
    spans, cur, start = [], "N", 0
    for s, aux in zip(ann.sample, ann.aux_note, strict=True):
        aux = (aux or "").strip("\x00").strip()
        if aux.startswith("("):
            spans.append((start, int(s), cur))
            cur, start = aux[1:], int(s)
    spans.append((start, n, cur))
    return [sp for sp in spans if sp[1] > sp[0]]


def max_v_run(sym: list[str]) -> int:
    best = run = 0
    for s in sym:
        run = run + 1 if s == "V" else 0
        best = max(best, run)
    return best


def windows(path: str):
    """(class, s0, truth) for one record, capped per class."""
    import wfdb

    ann = wfdb.rdann(path, "atr")
    hdr = wfdb.rdheader(path)
    fs, n_all = float(hdr.fs), int(hdr.sig_len)
    w = int(WIN_S * fs)
    spans = rhythm_spans(ann, n_all)
    beats = [(int(s), y) for s, y in zip(ann.sample, ann.symbol, strict=True) if y in BEATS]
    cnt: Counter = Counter()
    out = []

    def rhythm_at(a: int, b: int) -> set[str]:
        return {r for x, y, r in spans if x < b and y > a}

    # vt_full: consecutive windows inside each (VT span
    for a, b, r in spans:
        if r != "VT":
            continue
        for s0 in range(a, b - w + 1, w):
            if cnt["vt_full"] >= CAP["vt_full"]:
                break
            sym = [y for s, y in beats if s0 <= s < s0 + w]
            v = [s for s, y in beats if s0 <= s < s0 + w and y == "V"]
            hr = 60.0 * fs / float(np.median(np.diff(v))) if len(v) >= 3 else None
            cnt["vt_full"] += 1
            out.append(("vt_full", s0, {"v": sym.count("V"), "n": len(sym), "run": max_v_run(sym), "vt_hr": hr}))
    for s0 in range(int(fs), n_all - w, w):
        sym = [y for s, y in beats if s0 <= s < s0 + w]
        rh = rhythm_at(s0, s0 + w)
        run = max_v_run(sym)
        if run >= 3 and rh != {"VT"}:
            kind = "vt_run"
        elif "V" not in sym and not rh & {"VT", "VFL", "VF", "IVR", "NOD"} and len(sym) >= 5:
            kind = "control"
        else:
            continue
        if cnt[kind] >= CAP[kind]:
            continue
        cnt[kind] += 1
        out.append((kind, s0, {"v": sym.count("V"), "n": len(sym), "run": run, "rhythm": sorted(rh)}))
    return out, fs


def run(a) -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.pipeline import run_pipeline

    k, nsh = (int(v) for v in a.shard.split("/"))
    paths = sorted((db, str(p)[:-4]) for db, d in DBS.items() for p in (ROOT / d).glob("*.hea"))
    rows = []
    for db, path in paths[k::nsh]:
        try:
            wins, fs = windows(path)
        except Exception as ex:
            print(f"skip {path}: {ex}", file=sys.stderr)
            continue
        for kind, s0, tr in wins:
            row = {"db": db, "rec": Path(path).name, "s0": s0, "kind": kind, **tr}
            try:
                res = run_pipeline(load(path, s0, int(WIN_S * fs)), mode="A")
                m = res.measurements
                row.update(codes=sorted(res.findings.codes()), alerts=[x.code for x in res.safety.alerts],
                           hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms)
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
            rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows))


def score(files: list[Path]) -> None:
    rows = [r for f in files for r in json.loads(f.read_text())]
    crit = {"ventricular_tachycardia", "wide_complex_tachycardia", "ventricular_fibrillation"}
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["db"])].append(r)
        by[(r["kind"], "all")].append(r)
    for key in sorted(by):
        rs = [r for r in by[key] if "codes" in r]
        if not rs:
            continue
        fam = sum(bool(set(r["codes"]) & VT_CODES) for r in rs)
        alert = sum(bool(set(r["alerts"]) & crit) for r in rs)
        each = Counter(c for r in rs for c in set(r["codes"]) & VT_CODES)
        print(f"{key[0]:8s} {key[1]:9s} n={len(rs):4d} family={fam:4d} ({fam / len(rs):.2f}) "
              f"alert={alert:4d} ({alert / len(rs):.2f}) {dict(each)}")
    miss = Counter(c for r in rows if r["kind"] == "vt_full" and "codes" in r and not set(r["codes"]) & VT_CODES
                   for c in r["codes"])
    print("vt_full missed — what was named instead:", miss.most_common(12))
    print("errors:", sum("error" in r for r in rows))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score([Path(p) for p in sys.argv[2:]])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
