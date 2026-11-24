"""SD-149: a 3-channel strip shot in parts, each part its own photo filling the frame (synthetic, the truth known).

A strip page (layout "3x4 strip") is rendered, cut at page level into ``n`` overlapping parts, and every part is degraded as
a separate photo (desk, perspective, noise, JPEG). The truth per part is which groups (and how many seconds of each) it shows.

    python tools/scanner/strip_parts.py <out dir> <strips> <parts>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GROUP_FIRST = ("I", "aVR", "V1", "V4")


def strip_page(seed: int, ppm: float = 12.0):
    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.sources import pick_preset, synthetic_signal
    from ecgscan.layout import get_layout

    rng = np.random.default_rng(seed)
    sig = synthetic_signal(pick_preset(bench_config()["sources"]["synthetic"]["adult"], rng), rng, seed, duration_s=16.0)
    return render_page(sig.leads_mv, sig.fs, get_layout("3x4 strip"), RenderSpec(px_per_mm=ppm, grid="red"))


def cut(page: np.ndarray, gt, n: int, overlap: float = 0.25) -> list[tuple[np.ndarray, dict]]:
    """``n`` parts of the page, each with the groups it shows: {group: seconds visible}."""
    w = page.shape[1]
    width = int(w / n * (1 + overlap)) if n > 1 else w
    xs = [int(i * (w - width) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
    out = []
    for x in xs:
        seen = {}
        for tr in gt.traces:
            if tr.lead not in GROUP_FIRST:
                continue
            x0 = tr.x0_px
            x1 = x0 + len(tr.samples_mv) / gt.fs * gt.speed_mm_s * gt.px_per_mm
            vis = max(0.0, min(x1, x + width) - max(x0, x)) / (gt.speed_mm_s * gt.px_per_mm)
            label_in = x <= x0 - 4 * gt.px_per_mm and x0 < x + width      # the label sits ~6 mm before the trace
            if vis > 0:
                seen[tr.lead] = {"seconds": round(vis, 2), "label_in": bool(label_in)}
        out.append((page[:, x:x + width].copy(), seen))
    return out


def photo(part: np.ndarray, seed: int) -> np.ndarray:
    from ecgscan.bench.degrade import DegradeSpec, degrade

    spec = DegradeSpec(background=0.08, perspective=0.02, rotation_deg=2.0, noise=2.0, vignette=0.1, jpeg_quality=90)
    img, _ = degrade(part, spec, seed=seed, px_per_mm=12.0)
    return img


def main(out: Path, strips: int, n: int) -> None:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for s in range(strips):
        page, gt = strip_page(700 + s)
        for i, (part, seen) in enumerate(cut(page, gt, n)):
            name = f"strip{s:02d}_part{i:02d}.jpg"
            cv2.imwrite(str(out / name), photo(part, 700 + 31 * s + i))
            truth.append({"file": name, "strip": s, "part": i, "groups": seen})
    (out / "truth.json").write_text(json.dumps(truth, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
