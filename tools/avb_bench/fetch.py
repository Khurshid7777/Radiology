"""D293: AV-block bench — Chapman-Shaoxing / Ningbo (PhysioNet ecg-arrhythmia 1.0.0): every record labelled 2AVB,
2AVB type I (Mobitz I), 3AVB or wandering atrial pacemaker (2 only), plus random sinus bradycardia / sinus rhythm
controls that are not in data/svt_bench. Header list reused from data/svt_bench/headers.json.

    python tools/avb_bench/fetch.py --out data/avb_bench [--controls 600] [--workers 16]

Open data (CC BY 4.0), git-ignored like every downloaded set.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svt_bench.fetch import BASE, get  # noqa: E402

TARGET = {"195042002": "2AVB", "54016002": "2AVB1", "28189009": "2AVB2", "27885002": "3AVB", "195101003": "WAP"}
CONF = {"426177001": "SB", "426783006": "SR"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--controls", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=293)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    heads = json.loads((root / "data/svt_bench/headers.json").read_text())
    used = {r["name"] for r in json.loads((root / "data/svt_bench/manifest.json").read_text())["records"]}
    rng = random.Random(a.seed)
    tgt = [h for h in heads if set(h["dx"]) & set(TARGET)]
    chosen = list(tgt)
    for k in CONF:
        pool = [h for h in heads if k in h["dx"] and not set(h["dx"]) & set(TARGET) and h["name"] not in used]
        rng.shuffle(pool)
        chosen += pool[: a.controls]
    for h in chosen:
        h["labels"] = sorted({TARGET[d] for d in h["dx"] if d in TARGET} | {CONF[d] for d in h["dx"] if d in CONF})
    print(f"[avb] targets {len(tgt)}, controls {len(chosen) - len(tgt)}", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)

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
    print(f"[avb] done {len(chosen)}", file=sys.stderr)


if __name__ == "__main__":
    main()
