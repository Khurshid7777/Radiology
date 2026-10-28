"""S2 (TZ 18.4) acceptance: a rhythm strip on two sheets is stitched with no RR break > 20 ms.

Usage: python reports/scanner/s2-multipage/run_multipage.py out.json [n] [jitter_ms]
A 16-s lead-II strip (synthetic, rate 55-110/min, beat-to-beat RR variation +-``jitter_ms``)
is cut into two sheets that overlap by 1.5-4.5 s, each photographed (the bench's photo
degradation) and read as a two-page study. Per study: the seam found or flagged; the largest
|RR read - RR true| over the beats around the seam; a "silent" error is a beat lost or doubled
with no SEAM_UNCERTAIN."""
import dataclasses
import json
import sys

import cv2
import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from ecgscan.bench.cases import bench_config
from ecgscan.bench.degrade import DegradeSpec, degrade
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.bench.sources import derive_limb_leads
from ecgscan.layout import get_layout
from ecgscan.multipage import r_peaks
from ecgscan.pipeline import ScanHints, scan

FS, TOTAL_S, PPM = 500.0, 16.0, 12.0
EDGE_S = 0.4         # beats this close to the strip's own start / end are left out of the count


def study(i: int, jitter: float, photo: dict):
    rng = np.random.default_rng(300 + i)
    hr = int(55 + (i * 5.5) % 56)
    lap = float(rng.uniform(1.5, 4.5))
    cut = float(rng.uniform(7.0, 9.0))
    a, b = (0.0, cut + lap / 2), (cut - lap / 2, TOTAL_S)
    s = generate_synthetic_ecg(heart_rate_bpm=hr, seed=400 + i, duration_s=TOTAL_S, rr_jitter_ms=jitter)
    mv = derive_limb_leads({k: s.lead(k).astype(np.float64) / 1000.0 for k in s.lead_names})
    spec = dataclasses.replace(get_layout("single"), panel_s=TOTAL_S)
    page, gt = render_page(mv, FS, spec, RenderSpec(px_per_mm=PPM))
    x0 = gt.traces[0].x0_px
    sheets = [np.ascontiguousarray(page[:, 0:int(x0 + a[1] * 25 * PPM)]),
              np.ascontiguousarray(page[:, int(x0 + b[0] * 25 * PPM):])]
    data = []
    for k, sh in enumerate(sheets):
        img, _ = degrade(sh, DegradeSpec(**photo), seed=500 + 2 * i + k, px_per_mm=PPM)
        data.append(cv2.imencode(".png", img)[1].tobytes())
    res = scan(data, hints=ScanHints(pages=True))
    out = {"i": i, "hr": hr, "overlap_true_s": round(lap, 3), "status": res.status, "warnings": res.warnings,
           "reasons": res.reject_reasons}
    if res.status == "rejected" or "II" not in res.leads:
        return out
    tp = r_peaks(mv["II"], FS) / FS
    tp = tp[(tp > EDGE_S) & (tp < TOTAL_S - EDGE_S)]           # the strip's own ends are not shown whole
    rp = r_peaks(res.leads["II"].array(), FS) / FS
    flagged = "SEAM_UNCERTAIN" in res.warnings
    out.update(overlap_found_s=res.pages[0].overlap_s, flagged=flagged)
    # The read RR sequence laid along the true one where it fits best: a beat lost or doubled
    # anywhere leaves no fit (mean |RR difference| > 25 ms over the rest of the record).
    rr_t, rr_r = np.diff(tp), np.diff(rp)
    fits = [(float(np.mean(np.abs(rr_r[j:j + n] - rr_t[o:o + n]))), o, j)
            for o in range(3) for j in range(3)
            for n in [min(len(rr_t) - o, len(rr_r) - j)] if n >= 5]
    if not fits or min(fits)[0] > 0.025:
        out["silent"] = not flagged
        return out
    _, o, j = min(fits)
    n = min(len(rr_t) - o, len(rr_r) - j)
    seam_rr = [k for k in range(n) if abs((tp[o + k] + tp[o + k + 1]) / 2 - cut) <= lap / 2 + 1.0]
    errs = np.abs(rr_r[j:j + n] - rr_t[o:o + n])
    out.update(seam_rr_err_ms=round(1000 * float(errs[seam_rr].max()), 1) if seam_rr else None,
               other_rr_err_ms=round(1000 * float(np.delete(errs, seam_rr).max()), 1) if len(seam_rr) < n else None,
               silent=False)
    return out


def main(path: str, n: int, jitter: float) -> None:
    photo = bench_config()["slices"]["photo_clean"]["degrade"]
    rows = []
    for i in range(n):
        r = study(i, jitter, photo)
        rows.append(r)
        print(json.dumps(r), flush=True)
    stitched = [r for r in rows if r.get("seam_rr_err_ms") is not None and not r.get("flagged") and not r.get("silent")]
    summary = {"n": n, "jitter_ms": jitter,
               "stitched_unflagged": len(stitched),
               "flagged": sum(bool(r.get("flagged")) for r in rows),
               "rejected": sum(r["status"] == "rejected" for r in rows),
               "silent_errors": sum(bool(r.get("silent")) for r in rows),
               "max_seam_rr_err_ms": max((r["seam_rr_err_ms"] for r in stitched), default=None),
               "within_20ms": sum(r["seam_rr_err_ms"] <= 20 for r in stitched)}
    with open(path, "w") as f:
        json.dump({"summary": summary, "studies": rows}, f, indent=1, default=int)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20, float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
