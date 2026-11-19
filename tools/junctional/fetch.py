"""D329: Chapman 'AV junctional rhythm' (SNOMED 29320008) records no bench has taken. Open data (CC BY 4.0), git-ignored.

    python tools/junctional/fetch.py --out data/junctional_bench
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svt_bench.fetch import BASE, get

CODE = "29320008"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    heads = json.loads((root / "data/svt_bench/headers.json").read_text())
    used = set()
    for d in ("svt_bench", "avb_bench", "slowvt_bench", "untaken_bench"):
        p = root / "data" / d / "manifest.json"
        if p.exists():
            used |= {r["name"] for r in json.loads(p.read_text())["records"]}
    chosen = [dict(h, labels=["AVJR"]) for h in heads if CODE in h["dx"] and h["name"] not in used]
    print(f"[junctional] {len(chosen)} records", file=sys.stderr)
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
