"""Training data for the learned trace segmenter (TZ §14 M6, SD-051).

The segmenter sees the page after rectification (``geometry.working_px_per_mm``), so
samples are made in that space: a page rendered at a random scale, degraded by
everything rectification does not undo (paper, light, optics, sensor, JPEG; a small
residual crumple), then resampled to the working scale.

The mask is the true pen stroke: the page is rendered twice more without any
photometric damage — once as printed, once with an invisible pen (``trace_gray=255``,
text kept by ``text_gray``) — and both are warped by the same crumple (same seed,
the warp draws first); the strokes are where the two differ. A mask taken after blur
and JPEG (the first model, SD-052) taught the network the blurred, wider stroke:
QRS came out 12 ms wide. Traces and calibration pulses are foreground; grid, text
and paper are not.

Seeds come from their own range (``SEED_BASE``) and only synthetic records are used:
the benchmark's cases (and its PTB-XL records) never appear in training.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ecgscan.bench.cases import bench_config
from ecgscan.bench.degrade import DegradeSpec, degrade
from ecgscan.bench.render import GRID_PALETTES, RenderSpec, render_page
from ecgscan.bench.sources import pick_preset, synthetic_signal
from ecgscan.config import load_yaml
from ecgscan.layout import get_layout

SEED_BASE = 900_000_000
LAYOUTS = (("3x4+rhythm", 6), ("6x2", 2), ("12x1", 2))


def _pick(rng: np.random.Generator, options) -> object:
    names, weights = zip(*options, strict=True)
    w = np.array(weights, dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def sample_page(i: int, *, work_ppm: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Page ``i`` in the segmenter's input space: (BGR uint8, stroke mask uint8 0/1, meta)."""
    import cv2

    work_ppm = work_ppm or load_yaml("pipeline")["geometry"]["working_px_per_mm"]
    seed = SEED_BASE + i
    rng = np.random.default_rng(seed)
    src_cfg = bench_config()["sources"]["synthetic"]
    group = "pediatric" if rng.random() < 0.25 else "adult"
    sig = synthetic_signal(pick_preset(src_cfg[group], rng), rng, seed % 100000)
    layout = get_layout(_pick(rng, LAYOUTS))
    ppm = float(rng.uniform(work_ppm * 0.85, work_ppm * 1.5))
    spec = RenderSpec(
        px_per_mm=ppm,
        speed_mm_s=float(rng.choice([25, 25, 50])),
        gain_mm_mv=float(rng.choice([5, 10, 10, 10, 20])),
        grid=str(rng.choice(list(GRID_PALETTES))),
        trace_mm=float(rng.uniform(0.18, 0.5)),            # thin printers to thick pens
        trace_gray=int(rng.choice([15, 25, 40, 70, 110], p=[0.3, 0.3, 0.2, 0.12, 0.08])),   # incl. faded thermal
        grid_fade=float(rng.choice([0.0, 0.0, 0.3, 0.6])),
        header=bool(rng.random() < 0.8),
        labels=bool(rng.random() < 0.9),
        column_markers=bool(rng.random() < 0.3),
    )
    deg = DegradeSpec(
        crumple=float(rng.choice([0.0, 0.0, 0.4, 1.0])),
        creases=int(rng.choice([0, 0, 1, 2])),
        stains=int(rng.choice([0, 0, 1, 3])),
        glare=float(rng.choice([0.0, 0.0, 0.0, 0.6, 0.9])),
        shadow=float(rng.choice([0.0, 0.0, 0.4, 0.7])),
        vignette=float(rng.choice([0.0, 0.3, 0.6])),
        low_light=float(rng.choice([0.0, 0.0, 0.0, 0.4, 0.7])),
        noise=float(rng.choice([0.0, 2.0, 5.0, 9.0])),
        blur_px=float(rng.choice([0.0, 0.0, 0.6, 1.2, 2.0])) * ppm / 10.0,
        jpeg_quality=int(rng.choice([95, 85, 70, 55])),
    )
    shown, _ = render_page(sig.leads_mv, sig.fs, layout, spec)
    pen = replace(spec, trace_gray=0, grid_fade=1.0)         # the stroke itself, full contrast
    drawn, _ = render_page(sig.leads_mv, sig.fs, layout, pen)
    hidden, _ = render_page(sig.leads_mv, sig.fs, layout, replace(pen, trace_gray=255, text_gray=0))
    a, _ = degrade(shown, deg, seed=seed, px_per_mm=ppm)
    geom = DegradeSpec(crumple=deg.crumple)                  # the same warp, nothing else
    d, _ = degrade(drawn, geom, seed=seed, px_per_mm=ppm)
    b, _ = degrade(hidden, geom, seed=seed, px_per_mm=ppm)
    f = work_ppm / ppm
    size = (int(round(a.shape[1] * f)), int(round(a.shape[0] * f)))
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    d = cv2.resize(d, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    gd = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Covered by at least half of a pen pixel after resampling: the pen's own width.
    mask = ((gb - gd) > 0.5 * np.maximum(gb, 1.0)).astype(np.uint8)
    meta = {"seed": seed, "layout": layout.name, "ppm": round(ppm, 3), "render": spec.to_dict(),
            "degrade": deg.to_dict(), "ink_share": round(float(mask.mean()), 5)}
    return a, mask, meta


def crops(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, *, n: int, size: int = 256,
          ink_bias: float = 0.8) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` square crops; ``ink_bias`` of them centred on a stroke pixel (strokes are ~2 % of a page)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    out = []
    for _ in range(n):
        if ys.size and rng.random() < ink_bias:
            k = int(rng.integers(ys.size))
            cy, cx = int(ys[k]), int(xs[k])
            y0 = int(np.clip(cy - rng.integers(size), 0, h - size))
            x0 = int(np.clip(cx - rng.integers(size), 0, w - size))
        else:
            y0, x0 = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
        out.append((img[y0:y0 + size, x0:x0 + size].copy(), mask[y0:y0 + size, x0:x0 + size].copy()))
    return out


def build_shard(path: str | Path, start: int, count: int, *, crops_per_page: int = 12, size: int = 256) -> dict:
    """Pages ``start .. start+count-1`` as one ``.npz`` shard of crops (x: N×H×W×3 BGR, y: N×H×W)."""
    xs, ys, inks = [], [], []
    for i in range(start, start + count):
        img, mask, meta = sample_page(i)
        rng = np.random.default_rng(SEED_BASE + 7 * i + 1)
        for x, y in crops(img, mask, rng, n=crops_per_page, size=size):
            xs.append(x)
            ys.append(y)
        inks.append(meta["ink_share"])
    np.savez_compressed(path, x=np.stack(xs), y=np.stack(ys))
    return {"pages": count, "crops": len(xs), "ink_share_median": float(np.median(inks))}
