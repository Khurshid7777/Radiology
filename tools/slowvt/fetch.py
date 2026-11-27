"""D312: slow ventricular rhythm bench — Chapman-Shaoxing / Ningbo (PhysioNet ecg-arrhythmia 1.0.0): every record
labelled accelerated idioventricular rhythm (AIVR, SNOMED 61277005), plus wide-QRS controls that must NOT be called
ventricular: sinus tachycardia with RBBB or LBBB, and sinus rhythm with RBBB / LBBB. Not in data/svt_bench or avb_bench.

    python tools/slowvt/fetch.py --out data/slowvt_bench [--controls 150]

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
from svt_bench.fetch import BASE, get

TARGET = {"61277005": "AIVR"}
ST, SR, RBBB, LBBB = "427084000", "426783006", "59118001", "164909002"
CONTROLS = {"ST_BBB": lambda dx: ST in dx and (RBBB in dx or LBBB in dx),
            "SR_BBB": lambda dx: SR in dx and (RBBB in dx or LBBB in dx)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--controls", type=int, default=150)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=312)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    heads = json.loads((root / "data/svt_bench/headers.json").read_text())
    used = set()
    for d in ("svt_bench", "avb_bench"):
        p = root / "data" / d / "manifest.json"
        if p.exists():
            used |= {r["name"] for r in json.loads(p.read_text())["records"]}
    rng = random.Random(a.seed)
    chosen = [dict(h, labels=["AIVR"]) for h in heads if set(h["dx"]) & set(TARGET)]
    for name, sel in CONTROLS.items():
        pool = [h for h in heads if sel(set(h["dx"])) and not set(h["dx"]) & set(TARGET) and h["name"] not in used]
        rng.shuffle(pool)
        chosen += [dict(h, labels=[name]) for h in pool[: a.controls]]
    print(f"[slowvt] AIVR {sum(h['labels'] == ['AIVR'] for h in chosen)}, controls "
          f"{sum(h['labels'] != ['AIVR'] for h in chosen)}", file=sys.stderr)
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
    print(f"[slowvt] done {len(chosen)}", file=sys.stderr)


if __name__ == "__main__":
    main()
