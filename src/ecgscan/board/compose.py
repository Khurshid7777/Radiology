"""One zone from several photos of the same board (SD-066).

Every photo is mapped into board millimetres by its own markers, so the parts are
already registered: no feature matching, no guessing of overlaps. Each pixel of the
zone is taken from the photo that sees it at the highest resolution; the zone's
resolution is then that best value at its worst point, and a part of the zone no
photo covers is reported, never filled in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.board.detect import BoardFit, scale_map, warp_zone


@dataclass
class Composite:
    bgr: np.ndarray
    source: np.ndarray               # index of the photo each pixel came from (-1: none)
    coverage: float                  # fraction of the zone seen by at least one photo
    min_px_per_mm: float             # worst best-resolution over the covered zone
    mean_px_per_mm: float


def compose_zone(photos: list[np.ndarray], fits: list[BoardFit], px_per_mm: float) -> Composite:
    if not photos or len(photos) != len(fits) or len({f.spec.name for f in fits}) != 1:
        raise ValueError("compose_zone: one fit per photo, all of the same board")
    best = None
    out = None
    src = None
    for k, (img, fit) in enumerate(zip(photos, fits, strict=True)):
        zone, mask = warp_zone(img, fit, px_per_mm, with_mask=True)
        scale = scale_map(fit, px_per_mm)
        np.copyto(scale, np.float32(-1.0), where=~mask)
        if best is None:
            best, out = scale, zone
            src = np.full(mask.shape, -1, np.int8)
            np.copyto(src, np.int8(k), where=mask)
            continue
        # Masked copies, not fancy indexing: the zone is ~7 Mpx x 3 (0.9 s -> ~0.2 s, SD-069).
        take = scale > best
        np.copyto(out, zone, where=take[..., None])
        np.copyto(src, np.int8(k), where=take)
        np.maximum(best, scale, out=best)
    covered = src >= 0
    vals = best[covered]
    return Composite(out, src, float(covered.mean()), float(vals.min()) if vals.size else 0.0,
                     float(vals.mean()) if vals.size else 0.0)
