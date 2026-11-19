"""D317: Chapman records with labels no bench has taken yet (junctional, unspecified AV block, short PR, sinus arrest,
SA block, 426183003, atrial rhythm). Header list reused from data/svt_bench/headers.json; records already in other
benches are skipped. Open data (CC BY 4.0), git-ignored.

    python tools/untaken/fetch.py --out data/untaken_bench
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

TARGET = {"426664006": "AJR", "426648003": "JT", "233917008": "AVB", "49578007": "SPRI", "5609005": "SARREST",
          "65778007": "SAB", "426183003": "MOBITZ2?", "17366009": "ATRIAL_RHYTHM"}
CAP = {"AVB": 150}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    heads = json.loads((root / "data/svt_bench/headers.json").read_text())
    used = set()
    for d in ("svt_bench", "avb_bench", "slowvt_bench"):
        p = root / "data" / d / "manifest.json"
        if p.exists():
            used |= {r["name"] for r in json.loads(p.read_text())["records"]}
    rng = random.Random(317)
    chosen = []
    for code, lab in TARGET.items():
        pool = [h for h in heads if code in h["dx"] and h["name"] not in used]
        rng.shuffle(pool)
        pool = pool[: CAP.get(lab, len(pool))]
        for h in pool:
            if h["name"] not in {c["name"] for c in chosen}:
                chosen.append(dict(h, labels=sorted({TARGET[d] for d in h["dx"] if d in TARGET})))
    print(f"[untaken] {len(chosen)} records", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)

    def dl(h):
        d = a.out / h["folder"]; d.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".mat"):
            t = d / (h["name"] + ext)
            if not t.exists():
                t.write_bytes(get(BASE + h["folder"] + h["name"] + ext))

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(dl, chosen))
    (a.out / "manifest.json").write_text(json.dumps({"source": BASE, "records": chosen}, ensure_ascii=False))


if __name__ == "__main__":
    main()
