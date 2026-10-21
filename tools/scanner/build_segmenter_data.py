"""Build the M6 training/validation shards (SD-051).

    python tools/scanner/build_segmenter_data.py --out data/segmenter --pages 320 --val 32 --workers 4
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ecgscan.segmentation.dataset import build_shard


def _one(args):
    path, start, count = args
    return path, build_shard(path, start, count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/segmenter")
    ap.add_argument("--pages", type=int, default=320)
    ap.add_argument("--val", type=int, default=32, help="pages kept for validation (after the training pages)")
    ap.add_argument("--per-shard", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, first, n in (("train", 0, a.pages), ("val", a.pages, a.val)):
        for s in range(first, first + n, a.per_shard):
            p = out / f"{split}_{s:05d}.npz"
            if not p.exists():
                jobs.append((str(p), s, min(a.per_shard, first + n - s)))
    with ProcessPoolExecutor(a.workers) as ex:
        for path, info in ex.map(_one, jobs):
            print(path, json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
