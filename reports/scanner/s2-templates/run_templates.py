"""S2 (TZ 18.5) template detection on synthetic photos: is the layout found without a hint?

Usage: python reports/scanner/s2-templates/run_templates.py out.json [n_per_template]
Each template is rendered from synthetic 12-lead records (varied rate and seed), degraded
like the bench's photo slices (``bench.yaml`` ``_photo``, plus stains for half of them) and
scanned without a layout hint. "Wrong" = another layout named without LAYOUT_UNCERTAIN."""
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
from ecgscan.pipeline import scan

TEMPLATES = {"3x4+rhythm": {}, "3x4+3rhythm": {}, "6x2": {}, "12x1": {}, "single": {}, "3x4 strip": {"header": False}}


def main(out: str, n: int) -> None:
    photo = bench_config()["slices"]["photo_clean"]["degrade"]
    rows, summary = [], {}
    for name, render in TEMPLATES.items():
        ok = wrong = asked = 0
        for i in range(n):
            s = generate_synthetic_ecg(heart_rate_bpm=int(55 + 9 * i), seed=100 + i,
                                       duration_s=max(10.0, get_layout(name).record_s))
            mv = derive_limb_leads({k: s.lead(k).astype(np.float64) / 1000.0 for k in s.lead_names})
            page, _ = render_page(mv, 500.0, get_layout(name), RenderSpec(px_per_mm=12.0, **render))
            spec = DegradeSpec(**{**photo, **({"stains": 3} if i % 2 else {})})
            img, _ = degrade(page, spec, seed=200 + i, px_per_mm=12.0)
            res = scan(cv2.imencode(".png", img)[1].tobytes())
            got, unsure = res.layout.name, "LAYOUT_UNCERTAIN" in res.warnings
            ok += got == name
            wrong += got not in (None, name) and not unsure
            asked += got != name and (got is None or unsure)
            rows.append({"template": name, "i": i, "got": got, "uncertain": unsure, "status": res.status,
                         "reasons": res.reject_reasons})
            print(name, i, got, unsure, res.status, res.reject_reasons, flush=True)
        summary[name] = {"n": n, "correct": ok, "wrong_confident": wrong, "asked_or_unknown": asked}
    tot = sum(v["n"] for v in summary.values())
    summary["all"] = {"n": tot, "correct": sum(v["correct"] for v in summary.values()),
                      "wrong_confident": sum(v["wrong_confident"] for v in summary.values())}
    summary["all"]["correct_share"] = round(summary["all"]["correct"] / tot, 4)
    with open(out, "w") as f:
        json.dump({"summary": summary, "cases": rows}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
