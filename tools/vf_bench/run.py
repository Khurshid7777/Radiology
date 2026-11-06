"""D204: ventricular fibrillation bench — MIT-BIH VFDB and CUDB 10-s windows plus the PTB-XL bench.

    python tools/vf_bench/run.py fetch --out data/vf_bench          # open data (ODC-By), git-ignored
    python tools/vf_bench/run.py windows --data data/vf_bench --out <file.jsonl>
    python tools/vf_bench/run.py ptbxl --data data/ptbxl_rhythm --out <file.jsonl>
    python tools/vf_bench/run.py score <windows.jsonl> <ptbxl.jsonl>

A window is labelled by the rhythm it lies in entirely: VF / VFL (vfdb "(VF", "(VFL"; cudb between
"[" and "]"), VT ("(VT"), other organised rhythms, NOISE, ASYS. Per lead the detector's metrics are
kept, so the thresholds can be scored without re-reading the signals.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cardiolens.arrhythmia.vf import detect_vf, lead_is_vf

BASE = "https://physionet-open.s3.amazonaws.com/{db}/1.0.0/"
WIN_S = 10.0


def fetch(out: Path) -> None:
    for db in ("vfdb", "cudb"):
        d = out / db
        d.mkdir(parents=True, exist_ok=True)
        recs = urllib.request.urlopen(BASE.format(db=db) + "RECORDS", timeout=60).read().decode().split()
        for r in recs:
            for ext in ("hea", "dat", "atr"):
                t = d / f"{r}.{ext}"
                if not t.exists():
                    t.write_bytes(urllib.request.urlopen(BASE.format(db=db) + f"{r}.{ext}", timeout=120).read())


def _segments(db: str, ann, n: int) -> list[tuple[int, int, str]]:
    """(start, end, rhythm) spans of one record."""
    spans: list[tuple[int, int, str]] = []
    if db == "vfdb":
        cur, start = "N", 0
        for s, aux in zip(ann.sample, ann.aux_note, strict=True):
            aux = aux.strip("\x00").strip()
            if aux.startswith("("):
                spans.append((start, int(s), cur))
                cur, start = aux[1:], int(s)
        spans.append((start, n, cur))
    else:   # cudb: "[" ... "]" bound VF/VFL; elsewhere organised rhythm (or noise "~")
        cur, start = "N", 0
        for s, sym in zip(ann.sample, ann.symbol, strict=True):
            if sym == "[":
                spans.append((start, int(s), cur))
                cur, start = "VF", int(s)
            elif sym == "]":
                spans.append((start, int(s), cur))
                cur, start = "N", int(s)
        spans.append((start, n, cur))
    return [sp for sp in spans if sp[1] > sp[0]]


def windows(data: Path, out: Path) -> None:
    import wfdb

    with out.open("w") as fh:
        for db in ("vfdb", "cudb"):
            for hea in sorted((data / db).glob("*.hea")):
                rec = wfdb.rdrecord(str(hea.with_suffix("")))
                ann = wfdb.rdann(str(hea.with_suffix("")), "atr")
                fs = float(rec.fs)
                x = rec.p_signal.T * 1000.0           # mV -> µV
                w = int(WIN_S * fs)
                for a, b, rhythm in _segments(db, ann, x.shape[1]):
                    for s in range(a, b - w + 1, w):
                        seg = x[:, s:s + w]
                        if not np.isfinite(seg).all():
                            continue
                        vf, mts = detect_vf(seg, fs)
                        fh.write(json.dumps({"db": db, "rec": hea.stem, "start": s, "rhythm": rhythm,
                                             "vf": vf, "leads": mts}) + "\n")


def ptbxl(data: Path, out: Path) -> None:
    import wfdb

    man = json.loads((data / "manifest.json").read_text())["records"]
    with out.open("w") as fh:
        for r in man:
            rec = wfdb.rdrecord(str(data / r["filename_hr"]))
            vf, mts = detect_vf(rec.p_signal.T * 1000.0, float(rec.fs))
            fh.write(json.dumps({"ecg_id": r["ecg_id"], "labels": r["labels"], "vf": vf, "leads": mts}) + "\n")


def _rule(leads, **thr) -> bool:
    import cardiolens.arrhythmia.vf as V

    saved = {k: getattr(V, k) for k in thr}
    try:
        for k, v in thr.items():
            setattr(V, k, v)
        usable = [m for m in leads if m is not None]
        hits = sum(lead_is_vf(m) for m in usable)
        return bool(usable and hits >= 1 and hits / len(usable) >= V.VF_LEAD_SHARE_MIN)
    finally:
        for k, v in saved.items():
            setattr(V, k, v)


def score(win_file: Path, ptb_file: Path, **thr) -> dict:
    wins = [json.loads(line) for line in win_file.open()]
    ptb = [json.loads(line) for line in ptb_file.open()]
    by: dict[str, list[bool]] = {}
    for w in wins:
        key = "VF/VFL" if w["rhythm"] in ("VF", "VFL") else w["rhythm"]
        by.setdefault(key, []).append(_rule(w["leads"], **thr))
    out = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
    out["PTB-XL (all false)"] = f"{sum(_rule(p['leads'], **thr) for p in ptb)}/{len(ptb)}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "windows", "ptbxl", "score"])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.out)
    elif a.cmd == "windows":
        windows(a.data, a.out)
    elif a.cmd == "ptbxl":
        ptbxl(a.data, a.out)
    else:
        print(json.dumps(score(*a.files), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
