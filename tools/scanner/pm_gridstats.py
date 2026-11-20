"""R8 (SD-121): aggregate darkness statistics of the PM-ECG-ID photographs (no per-image values are kept in git).

Only class-level numbers are reported (per source folder and overall quantiles): how dark the printed grid is next to the
trace and the paper. They calibrate the synthetic renderer's palette; the scanner is never tuned on single images.

    python tools/scanner/pm_gridstats.py --root data/pmecgid --out <dir> [--shard k/n] [--every 3]
    python tools/scanner/pm_gridstats.py --aggregate <dir> [<dir> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PHOTOS = ("photos_bents", "photos_crumbles", "photos_doogee", "photos_iphone", "photos_samsung", "photos_scans",
          "photos_screens")


def page_stats(gray: np.ndarray, px_per_mm: float) -> dict | None:
    """Paper level, trace contrast, grid-line contrast and their ratio on a rectified grey page."""
    h, w = gray.shape
    y0, y1, x0, x1 = int(0.1 * h), int(0.85 * h), int(0.1 * w), int(0.9 * w)
    core = gray[y0:y1, x0:x1].astype(np.float32)
    if core.size == 0:
        return None
    paper = float(np.percentile(core, 85))
    trace = float(np.percentile(core, 0.5))
    colmed = np.median(core, axis=0)                      # the sparse trace drops out; vertical grid lines stay as dips
    dip, between = float(np.percentile(colmed, 5)), float(np.percentile(colmed, 50))
    rowmed = np.median(core, axis=1)
    rdip, rbetween = float(np.percentile(rowmed, 5)), float(np.percentile(rowmed, 50))
    trace_c = paper - trace
    if trace_c < 20:
        return None
    grid_c = max(between - dip, rbetween - rdip)
    return {"paper": paper, "trace_contrast": trace_c, "grid_contrast": float(grid_c),
            "grid_over_trace": float(grid_c / trace_c), "ppm": px_per_mm}


def run(root: Path, out: Path, shard: str, every: int) -> None:
    import cv2

    from ecgscan.pipeline import ScanDebug, ScanHints, scan

    meta = list(csv.DictReader((root / "final_data" / "metadata.csv").open()))
    rels = sorted(r["Image relative path"] for r in meta
                  if r["Image relative path"].split("/")[0] in PHOTOS and r["ECG number of pages"] == "1")
    k, n = (int(v) for v in shard.split("/"))
    rels = rels[::every][k::n]
    out.mkdir(parents=True, exist_ok=True)
    folder_of = {r["Image relative path"]: r["Image relative path"].split("/")[0] for r in meta}
    fmt_of = {r["Image relative path"]: r["ECG format"] for r in meta}
    with (out / "stats.jsonl").open("w") as f:
        for rel in rels:
            dbg = ScanDebug()
            try:
                scan(str(root / "final_data" / "visual_data" / rel), debug=dbg,
                     hints=ScanHints(paper_speed_mm_s=25, gain_mm_mv=10))
            except Exception as exc:        # a page the scanner cannot process gives no statistic
                print(f"skipped {rel}: {type(exc).__name__}", flush=True)
                continue
            if dbg.rectified is None:
                continue
            gray = cv2.cvtColor(dbg.rectified.bgr, cv2.COLOR_BGR2GRAY)
            s = page_stats(gray, float(dbg.rectified.px_per_mm))
            if s:
                s.update({"folder": folder_of[rel], "format": fmt_of[rel]})
                f.write(json.dumps(s) + "\n")
                f.flush()


def aggregate(dirs: list[Path]) -> str:
    rows = [json.loads(x) for d in dirs for x in (d / "stats.jsonl").read_text().splitlines() if x.strip()]
    lines = [f"pages: {len(rows)}"]

    def q(vals):
        a = np.array(vals)
        return " ".join(f"p{p}={np.percentile(a, p):.2f}" for p in (10, 25, 50, 75, 90))

    for key in ("grid_over_trace", "trace_contrast", "grid_contrast", "paper"):
        lines.append(f"{key}: {q([r[key] for r in rows])}")
    for fld in sorted({r["folder"] for r in rows}):
        sub = [r for r in rows if r["folder"] == fld]
        lines.append(f"  {fld}: n={len(sub)} grid_over_trace median {np.median([r['grid_over_trace'] for r in sub]):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--aggregate", nargs="+", type=Path)
    a = ap.parse_args()
    if a.aggregate:
        print(aggregate(a.aggregate))
    else:
        run(a.root, a.out, a.shard, a.every)
