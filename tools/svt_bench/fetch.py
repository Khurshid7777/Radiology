"""D268: SVT bench — Chapman-Shaoxing / Ningbo (PhysioNet ecg-arrhythmia 1.0.0) records labelled SVT, AVNRT,
AVRT, atrial or paroxysmal SVT, with sinus tachycardia, flutter and sinus rhythm as confounders.

    python tools/svt_bench/fetch.py --out data/svt_bench [--controls 300] [--workers 16]

Every header is read (45k x ~1 KB, S3 mirror), the selection is written to manifest.json, and only the
selected .hea / .mat pairs are downloaded. Licensed data: git-ignored, never committed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/1.0.0/"
SVT = {"426761007": "SVT", "233896004": "AVNRT", "233897008": "AVRT", "713422000": "AT", "67198005": "PSVT"}
CONF = {"427084000": "STACH", "164890007": "AFL", "426783006": "SR"}


def get(url: str) -> bytes:
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == 5:
                raise
            time.sleep(1.5 * (i + 1))
    return b""


def header(folder: str, name: str) -> dict:
    txt = get(BASE + folder + name + ".hea").decode("utf-8", "replace")
    dx = re.search(r"#\s*Dx:\s*([0-9,\s]+)", txt)
    age = re.search(r"#\s*Age:\s*([0-9.]+)", txt)
    sex = re.search(r"#\s*Sex:\s*(\w+)", txt)
    return {"folder": folder, "name": name, "dx": [d.strip() for d in (dx.group(1) if dx else "").split(",") if d.strip()],
            "age": float(age.group(1)) if age else None, "sex": sex.group(1) if sex else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    hdr_cache = a.out / "headers.json"
    if hdr_cache.exists():
        heads = json.loads(hdr_cache.read_text())
    else:
        folders = [f.strip() for f in get(BASE + "RECORDS").decode().split() if f.strip()]
        with ThreadPoolExecutor(a.workers) as ex:
            names = list(ex.map(lambda f: (f, [n for n in get(BASE + f + "RECORDS").decode().split() if n]), folders))
        jobs = [(f, n) for f, ns in names for n in ns]
        print(f"[svt] {len(jobs)} headers", file=sys.stderr)
        heads = []
        with ThreadPoolExecutor(a.workers) as ex:
            for i, h in enumerate(ex.map(lambda j: header(*j), jobs)):
                heads.append(h)
                if i % 5000 == 0:
                    print(f"[svt] headers {i}", file=sys.stderr)
        hdr_cache.write_text(json.dumps(heads))
    rng = random.Random(a.seed)
    svt = [h for h in heads if set(h["dx"]) & set(SVT)]
    conf = {k: [h for h in heads if k in h["dx"] and not set(h["dx"]) & set(SVT)] for k in CONF}
    chosen = list(svt)
    for hs in conf.values():
        rng.shuffle(hs)
        chosen += hs[: a.controls]
    for h in chosen:
        h["labels"] = sorted({SVT[d] for d in h["dx"] if d in SVT} | {CONF[d] for d in h["dx"] if d in CONF})
    print(f"[svt] SVT-family {len(svt)}, controls {len(chosen) - len(svt)}", file=sys.stderr)

    def dl(h: dict) -> None:
        d = a.out / h["folder"]
        d.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".mat"):
            t = d / (h["name"] + ext)
            if not t.exists():
                t.write_bytes(get(BASE + h["folder"] + h["name"] + ext))

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(dl, chosen))
    (a.out / "manifest.json").write_text(json.dumps({"source": BASE, "records": chosen}, ensure_ascii=False))
    print(f"[svt] done {len(chosen)}", file=sys.stderr)


if __name__ == "__main__":
    main()
