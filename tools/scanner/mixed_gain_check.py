"""SD-153: sheets with the chest leads at half the limb gain ("10/5 mm/mV"): the chest amplitude after the scan.

    python tools/scanner/mixed_gain_check.py <pages per layout>
"""

from __future__ import annotations

import json
import sys

import numpy as np


def sheet(seed: int, layout: str, *, chest: float | None, header_pair: bool):
    import cv2

    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.degrade import DegradeSpec, degrade
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.sources import pick_preset, synthetic_signal
    from ecgscan.layout import get_layout

    rng = np.random.default_rng(seed)
    sig = synthetic_signal(pick_preset(bench_config()["sources"]["synthetic"]["adult"], rng), rng, seed)
    spec = RenderSpec(px_per_mm=12.0, gain_chest_mm_mv=chest, header_settings=True)
    page, gt = render_page(sig.leads_mv, sig.fs, get_layout(layout), spec)
    if chest and not header_pair:                       # the same sheet, the header saying one gain only
        spec2 = RenderSpec(px_per_mm=12.0, gain_chest_mm_mv=chest, header_settings=False)
        page, gt = render_page(sig.leads_mv, sig.fs, get_layout(layout), spec2)
    img, _ = degrade(page, DegradeSpec(background=0.05, noise=1.5, jpeg_quality=92), seed=seed, px_per_mm=12.0)
    return cv2.imencode(".jpg", img)[1].tobytes(), gt


def r_peak_error(gt, result) -> dict:
    """Median relative error of the peak-to-peak amplitude, chest and limb leads."""
    out = {"chest": [], "limb": []}
    for tr in gt.traces:
        if tr.kind != "panel" or tr.lead not in result.leads:
            continue
        got = np.array([np.nan if v is None else v for v in result.leads[tr.lead].samples_mv], float)
        if not np.isfinite(got).any():
            continue
        true_pp = float(np.ptp(tr.samples_mv))
        if true_pp < 0.3:
            continue
        err = abs(float(np.nanmax(got) - np.nanmin(got)) - true_pp) / true_pp
        out["chest" if tr.lead.startswith("V") else "limb"].append(err)
    return {k: (round(float(np.median(v)), 3) if v else None) for k, v in out.items()}


def main(n: int) -> None:
    from ecgscan.pipeline import ScanHints, scan

    for layout in ("3x4+rhythm", "6x2"):
        for s in range(n):
            for case, chest, pair, hint in (("mixed_header", 5.0, True, None), ("mixed_no_header", 5.0, False, None),
                                            ("mixed_hint", 5.0, False, 5.0), ("plain", None, False, None)):
                data, gt = sheet(900 + s, layout, chest=chest, header_pair=pair)
                res = scan(data, hints=ScanHints(paper_speed_mm_s=25, gain_chest_mm_mv=hint))
                print(json.dumps({"layout": layout, "seed": s, "case": case, "status": res.status,
                                  "chest_gain": res.calibration.gain_chest_mm_mv, "gain": res.calibration.gain_mm_mv,
                                  "warn": [w for w in res.warnings if w in ("MIXED_GAIN", "GAIN_SOURCES_DISAGREE")],
                                  **r_peak_error(gt, res)}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
