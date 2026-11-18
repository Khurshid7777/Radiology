"""R8 (SD-111): the scanner against PM-ECG-ID — photographs and scans of 100 PTB-XL records with the true signal.

    python tools/scanner/pmecgid_bench.py --root data/pmecgid --set photos --out <dir> [--resume] [--shard k/n]
    python tools/scanner/pmecgid_bench.py --root data/pmecgid --set augmented --out <dir> --names <file>

The archive is test-only (never tuned on). ``--set photos``: the seven ``photos_*`` folders, single-page records;
``--set augmented``: the images listed in ``--names`` (``final_data/visual_data/<folder>/<image>`` per line).
The scanner gets no speed or gain hints: it reads them from the page, as in the bot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUT = {"6x2": "6x2", "3x4": "3x4", "3x4+1R": "3x4+rhythm", "3x4+3R": "3x4+3rhythm", "12x1": "12x1",
          "6x2+1R": "6x2+rhythm"}                    # 6x2 + 1R has no registry entry: windows of 6x2, the strip on top
FS = 500.0
PHOTOS = ("photos_bents", "photos_crumbles", "photos_doogee", "photos_iphone", "photos_samsung", "photos_scans",
          "photos_screens")


def _rhythm_names(rhythm: np.ndarray, cut: np.ndarray, window_s: float) -> list[str]:
    """Which lead each rhythm column is: the lead whose cutout it repeats (checked, not assumed)."""
    n = len(cut)
    out = []
    for j in range(rhythm.shape[1]):
        best = min((float(np.mean((rhythm[w * n:(w + 1) * n, j] - cut[:, li]) ** 2)), li)
                    for li in range(12) for w in range(len(rhythm) // n))
        out.append(LEADS[best[1]] if best[0] < 1e-9 else "")
    return out


def build_cases(root: Path, rel_paths: list[str]) -> tuple[list, list[dict]]:
    from ecgscan.bench.cases import Case
    from ecgscan.bench.render import PageGT, TraceGT
    from ecgscan.layout import get_layout

    data = root / "final_data"
    leads, rhythms = np.load(data / "data" / "leads.npz"), np.load(data / "data" / "rhythms.npz")
    meta = {r["Image relative path"]: r for r in csv.DictReader((data / "metadata.csv").open())}
    cases, skipped = [], []
    for rel in rel_paths:
        r = meta[rel]
        folder = rel.split("/")[0]
        if r["ECG number of pages"] != "1":
            skipped.append({"image": rel, "why": "multi-page record (not scored)"})
            continue
        path = data / "visual_data" / rel
        if not path.exists():
            skipped.append({"image": rel, "why": "not downloaded"})
            continue
        x, rh = leads[r["ECG ID"]].astype(np.float64), rhythms[r["ECG ID"]].astype(np.float64)
        fmt = r["ECG format"]
        lay = get_layout(LAYOUT[fmt])
        traces = []
        for lead, (t0, t1) in lay.lead_windows().items():
            seg = x[:, LEADS.index(lead)]
            if len(seg) != int(round((t1 - t0) * FS)):
                skipped.append({"image": rel, "why": f"cutout {len(seg)} samples, window {t1 - t0:.1f} s"})
                traces = None
                break
            traces.append(TraceGT(lead, "panel", t0, seg, 0.0, 0.0))
        if traces is None:
            continue
        names = _rhythm_names(rh, x, lay.panel_s) if rh.shape[1] else []
        if any(not n for n in names):
            skipped.append({"image": rel, "why": "rhythm strip not matched to a lead"})
            continue
        for j, name in enumerate(names):
            traces.append(TraceGT(name, "rhythm", 0.0, rh[:, j], 0.0, 0.0))
        truth = PageGT(layout=fmt, speed_mm_s=25.0, gain_mm_mv=10.0, px_per_mm=0.0, fs=FS, width_px=0,
                       height_px=0, header_box_px=None, traces=traces)
        cid = f"{folder.replace('augmentation_', '')}/{Path(rel).stem}"
        slice_name = ("real_" if folder in PHOTOS else "aug_") + folder.replace("augmentation_", "")
        cases.append(Case(cid, slice_name, path, truth,
                          {"case_id": cid, "slice": slice_name, "layout": fmt, "source": {"id": r["ECG ID"]},
                           "assumed": {}, "true_px_per_mm": None, "pm_format": fmt}))
    return cases, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/pmecgid"))
    ap.add_argument("--set", choices=("photos", "augmented"), required=True)
    ap.add_argument("--names", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--user-hints", action="store_true",
                    help="give the scanner the speed and gain a user would confirm (25 mm/s, 10 mm/mV; SD-112)")
    ap.add_argument("--user-layout", help="also give this layout, as a doctor would pick it (SD-115); use with --only "
                                          "for pages of that one format")
    ap.add_argument("--only", type=Path, help="file of case ids (folder/image stem), one per line: score just these")
    ap.add_argument("--wide-lag", type=float, default=None, help="also score r after this wider time-origin search, s")
    a = ap.parse_args()

    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.digitizers import default_digitizers
    from ecgscan.bench.run import run_bench

    meta = list(csv.DictReader((a.root / "final_data" / "metadata.csv").open()))
    if a.set == "photos":
        rels = [r["Image relative path"] for r in meta if r["Image relative path"].split("/")[0] in PHOTOS]
    else:
        rels = [n.split("visual_data/")[1] for n in a.names.read_text().split("\n") if n and "/photos_" not in n]
    if a.only:
        keep = {x.strip() for x in a.only.read_text().split("\n") if x.strip()}
        rels = [r for r in rels if r.rsplit(".", 1)[0] in keep]
    k, n = (int(v) for v in a.shard.split("/"))
    rels = sorted(rels)[k::n]
    cases, skipped = build_cases(a.root, rels)
    if a.limit:
        cases = cases[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    import json
    (a.out / "skipped.json").write_text(json.dumps(skipped, indent=1, ensure_ascii=False))
    print(f"{len(cases)} cases, {len(skipped)} skipped", flush=True)
    cfg = bench_config()
    if a.wide_lag:
        cfg["metrics"]["wide_lag_s"] = a.wide_lag
    digs = [d for d in default_digitizers(cfg) if d.name == "ecgscan"]
    if a.user_hints:
        digs[0].hints = {"paper_speed_mm_s": 25.0, "gain_mm_mv": 10.0}
    if a.user_layout:
        digs[0].hints = {**(digs[0].hints or {}), "layout": a.user_layout}
    run_bench("standard", a.out, digitizers=digs, cases=cases, resume=a.resume, config=cfg)


if __name__ == "__main__":
    main()
