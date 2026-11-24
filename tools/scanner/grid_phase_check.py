"""SD-143: the frame check's grid phase and zone agreement on synthetic frames with a known grid.

A rendered page (red grid, a 3x4+rhythm record) is mapped into a 1280 x 900 frame by a known transform: flat, turned 2-4
degrees, or seen with the camera tilted (the top of the frame 25 % narrower, about 15 degrees). The true 5-mm lines are
known, so the phase the check reports can be scored. Only synthetic data; nothing here tunes the check.

    python tools/scanner/grid_phase_check.py <out.json> <pages>
"""
from __future__ import annotations

import json
import math
import sys
import time

import cv2
import numpy as np

from ecgscan.bench.cases import bench_config
from ecgscan.bench.render import RenderSpec, render_page
from ecgscan.bench.sources import pick_preset, synthetic_signal
from ecgscan.capture_qc import check_frame
from ecgscan.layout import get_layout

PAGE_PPM = 8.0
FW, FH = 1280, 900

def page(seed, grid):
    rng = np.random.default_rng(seed)
    sig = synthetic_signal(pick_preset(bench_config()["sources"]["synthetic"]["adult"], rng), rng, seed)
    img, _ = render_page(sig.leads_mv, sig.fs, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=PAGE_PPM, grid=grid))
    return img

def frame(pg, scale, ox, oy, rot_deg, persp, rng):
    """The page mapped into the frame; returns (bgr, map page->frame as a 3x3 homography)."""
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    A = np.array([[scale * c, -scale * s, ox], [scale * s, scale * c, oy], [0, 0, 1.0]])
    H = A
    if persp:
        # camera tilted about the horizontal axis: the top of the frame narrower than the bottom
        src = np.float32([[0, 0], [FW, 0], [FW, FH], [0, FH]])
        t = persp / 2
        dst = np.float32([[FW * t, 0], [FW * (1 - t), 0], [FW, FH], [0, FH]])
        H = cv2.getPerspectiveTransform(src, dst) @ A
    bg = np.full((FH, FW, 3), (110, 125, 140), np.uint8)
    out = cv2.warpPerspective(pg, H, (FW, FH), dst=bg, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_AREA)
    out = np.clip(out.astype(np.float32) + rng.normal(0, 2.0, out.shape), 0, 255).astype(np.uint8)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    return out, H


def phase_error(q, H, axis: int) -> float:
    """How far (in parts of the period) the reported grid is from the true 5-mm lines on the frame's centre row (axis 0,
    vertical lines) or centre column (axis 1, horizontal lines)."""
    cx, cy = FW / 2, FH / 2
    u, v, w = np.linalg.inv(H) @ np.array([cx, cy, 1.0])
    u, v = u / w, v / w
    step = 5 * PAGE_PPM
    if axis == 0:                         # the page line u = const nearest the centre, where it crosses the row cy
        uu = math.floor(u / step) * step
        (x1, y1), (x2, y2) = [(p[0] / p[2], p[1] / p[2]) for p in (H @ np.array([uu, vv, 1.0]) for vv in (v - 50, v + 50))]
        d = x1 + (cy - y1) * (x2 - x1) / (y2 - y1) - (q.grid_phase_px[0] + q.grid_slope[0] * cy)
    else:
        vv = math.floor(v / step) * step
        (x1, y1), (x2, y2) = [(p[0] / p[2], p[1] / p[2]) for p in (H @ np.array([uu, vv, 1.0]) for uu in (u - 50, u + 50))]
        d = y1 + (cx - x1) * (y2 - y1) / (x2 - x1) - (q.grid_phase_px[1] + q.grid_slope[1] * cx)
    period = q.grid_period_px
    return abs(((d + period / 2) % period) - period / 2) / period


def measure(n_pages: int, seed: int = 1) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n_pages):
        pg = page(100 + i, "red")
        for kind in ("flat", "rot", "persp"):
            scale = float(rng.uniform(3.0, 6.0)) / PAGE_PPM          # 3-6 px/mm on the frame
            pw, ph = pg.shape[1] * scale, pg.shape[0] * scale
            ox, oy = (FW - pw) / 2 + rng.uniform(-30, 30), (FH - ph) / 2 + rng.uniform(-30, 30)
            rot = float(rng.uniform(2, 4) * rng.choice([-1, 1])) if kind == "rot" else 0.0
            bgr, H = frame(pg, scale, ox, oy, rot, 0.25 if kind == "persp" else 0.0, rng)
            t0 = time.perf_counter()
            q = check_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            dt = time.perf_counter() - t0
            r = {"kind": kind, "ppm": round(scale * PAGE_PPM, 2), "found": q.grid_phase_px is not None, "agree": q.grid_zones_agree,
                 "zones": q.grid_zones, "gper": q.grid_period_px, "dt": round(dt, 3), "check_ppm": q.px_per_mm}
            if q.grid_phase_px is not None and kind != "persp":
                r["ex"], r["ey"] = round(phase_error(q, H, 0), 3), round(phase_error(q, H, 1), 3)
            rows.append(r)
    return rows


if __name__ == "__main__":
    out = measure(int(sys.argv[2]))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f)
