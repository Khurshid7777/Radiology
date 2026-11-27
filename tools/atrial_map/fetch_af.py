"""D359: a NEW AF / ectopy bench from Chapman-Shaoxing / Ningbo (PhysioNet ecg-arrhythmia 1.0.0) for the acceptance of
D358's flutter guard — records never used by any earlier bench (svt, avb, untaken, junctional).

    python tools/atrial_map/fetch_af.py --out data/af_bench [--seed 359] [--workers 16]

Reuses the header list cached by tools/svt_bench (data/svt_bench/headers.json). Groups (each without the others'
rhythm codes where noted): AF 400, AFL 300, SR + PAC 300, SR + PVC 300, sinus arrhythmia 200. Licensed data:
git-ignored, never committed."""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "svt_bench"))
from fetch import BASE, get

AF, AFL, PAC, PVC, PVC2, SA = "164889003", "164890007", "284470004", "427172004", "17338001", "427393009"
GROUPS = {"AFIB": (lambda d: AF in d and AFL not in d, 400), "AFL": (lambda d: AFL in d and AF not in d, 300),
          "PAC": (lambda d: PAC in d and not d & {AF, AFL}, 300),
          "PVC": (lambda d: bool(d & {PVC, PVC2}) and not d & {AF, AFL}, 300),
          "SARRH": (lambda d: SA in d and not d & {AF, AFL, PAC, PVC, PVC2}, 200)}
USED_BENCHES = ("svt_bench", "avb_bench", "untaken_bench", "junctional_bench")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=359)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    heads = json.loads((ROOT / "data/svt_bench/headers.json").read_text())
    used = set()
    for b in USED_BENCHES:
        m = ROOT / "data" / b / "manifest.json"
        if m.exists():
            used |= {r["name"] for r in json.loads(m.read_text())["records"]}
    rng = random.Random(a.seed)
    chosen, taken = [], set()
    for lab, (pick, n) in GROUPS.items():
        pool = [h for h in heads if h["name"] not in used and h["name"] not in taken and pick(set(h["dx"]))]
        rng.shuffle(pool)
        for h in pool[:n]:
            chosen.append(dict(h, labels=[lab]))
            taken.add(h["name"])
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
    (a.out / "manifest.json").write_text(json.dumps({"source": BASE, "seed": a.seed, "records": chosen}))
    print(f"[af_bench] {len(chosen)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
