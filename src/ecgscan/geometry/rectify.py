"""Page rectification to a known, square px/mm (TZ §5.3).

1. The grid pitch is read on the photo as taken (``estimate_grid``) — this is the
   resolution check against the 8 px/mm floor;
2. the photo is resampled to ``working_px_per_mm`` (never upscaled): all later steps
   run at one known scale, and a 12 Mpx photo stops costing 10+ s;
3. perspective is removed by the CardioLens grid-texture homography (D64, with its
   quadrilateral and vanishing-point fallbacks) — reused, not rewritten (SD-003);
4. the grid is read again on the result and the image is scaled so that one
   millimetre is exactly ``working_px_per_mm`` pixels on both axes;
5. the residual: the pitch is read per tile; its spread across the page, times
   half a tile, is how far (mm) a grid line can sit from the ideal lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.geometry.grid import GridEstimate, estimate_grid


@dataclass
class Rectified:
    bgr: np.ndarray
    gray: np.ndarray
    px_per_mm: float                 # exact on the rectified image
    raw_grid: GridEstimate           # as photographed
    grid: GridEstimate               # after rectification (before the final exact scaling)
    residual_mm: float | None
    tile_pitches: list[float]


def _fast_grid(bgr: np.ndarray) -> GridEstimate:
    """Grid on a half-size copy when the photo is large (pitch is scale-relative)."""
    import cv2

    h, w = bgr.shape[:2]
    if max(h, w) <= 2400:
        return estimate_grid(bgr)
    small = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    g = estimate_grid(small)
    return GridEstimate(None if g.px_per_mm_x is None else 2 * g.px_per_mm_x,
                        None if g.px_per_mm_y is None else 2 * g.px_per_mm_y,
                        g.score_x, g.score_y, g.channel, g.confidence)


def tile_residual(bgr: np.ndarray, ppm: float, tiles: tuple[int, int] = (2, 3)) -> tuple[float | None, list[float]]:
    h, w = bgr.shape[:2]
    th, tw = h // tiles[0], w // tiles[1]
    vals = []
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            g = estimate_grid(bgr[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
            if g.px_per_mm is not None and g.confidence >= 0.5 and abs(g.px_per_mm - ppm) < 0.2 * ppm:
                vals.append(g.px_per_mm)
    if len(vals) < 3:
        return None, vals
    spread = float(np.std(vals) / np.mean(vals))
    return spread * (min(th, tw) / ppm) / 2.0, vals


def _dewarp(gray: np.ndarray, bgr: np.ndarray, fit_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """D64 homography fitted on a ``fit_scale`` copy, applied at full scale: H = S^-1 H_s S."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import dewarp_page

    if fit_scale >= 0.999:
        out = dewarp_page(gray)
        hm, size = dewarp_page.last_transform  # type: ignore[attr-defined]
    else:
        small = cv2.resize(gray, None, fx=fit_scale, fy=fit_scale, interpolation=cv2.INTER_AREA)
        dewarp_page(small)
        hm_s, size_s = dewarp_page.last_transform  # type: ignore[attr-defined]
        sc = np.diag([fit_scale, fit_scale, 1.0])
        hm = np.linalg.inv(sc) @ np.asarray(hm_s, dtype=np.float64) @ sc
        size = (int(round(size_s[0] / fit_scale)), int(round(size_s[1] / fit_scale)))
        if np.allclose(hm, np.eye(3)) and size == gray.shape[1::-1]:
            return gray, bgr
        out = cv2.warpPerspective(gray, hm, size, flags=cv2.INTER_LINEAR, borderValue=255)
    if np.allclose(hm, np.eye(3)) and tuple(size) == tuple(gray.shape[1::-1]):
        return out, bgr
    bgr_out = cv2.warpPerspective(bgr, np.asarray(hm, dtype=np.float64), tuple(int(v) for v in size),
                                  flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return out, bgr_out


def rectify(bgr: np.ndarray, *, working_px_per_mm: float, raw_grid: GridEstimate | None = None,
            max_upscale: float = 1.0, fit_px_per_mm: float | None = None) -> Rectified:
    import cv2

    from cardiolens.layer1_ingestion.digitize import estimate_skew

    raw = raw_grid or _fast_grid(bgr)
    if raw.px_per_mm is None:
        raise ValueError("no grid")
    s = min(max_upscale, working_px_per_mm / raw.px_per_mm)
    if abs(s - 1.0) > 1e-3:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fit_scale = 1.0 if not fit_px_per_mm else min(1.0, fit_px_per_mm / working_px_per_mm)
    gray_r, bgr_r = _dewarp(gray, bgr, fit_scale)
    # Rotation: the texture homography steps aside for affine cases (D64), so de-skew here
    # (pulse strokes and baselines are then axis-aligned for every later step).
    skew = estimate_skew(gray_r)
    if abs(skew) > 0.2:
        h, w = gray_r.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        gray_r = cv2.warpAffine(gray_r, m, (w, h), borderValue=255)
        bgr_r = cv2.warpAffine(bgr_r, m, (w, h), borderValue=(255, 255, 255))
    g = estimate_grid(bgr_r)
    if g.px_per_mm_x is None or g.px_per_mm_y is None:
        g = GridEstimate(raw.px_per_mm * s, raw.px_per_mm * s, raw.score_x, raw.score_y, raw.channel,
                         raw.confidence * 0.5)
    # Exact working scale on both axes (a few % either way after the homography; the
    # resolution check was made on the photo as taken).
    fx, fy = working_px_per_mm / g.px_per_mm_x, working_px_per_mm / g.px_per_mm_y
    bgr_r = cv2.resize(bgr_r, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    gray_r = cv2.resize(gray_r, (bgr_r.shape[1], bgr_r.shape[0]), interpolation=cv2.INTER_LINEAR)
    residual, tiles = tile_residual(bgr_r, working_px_per_mm)
    return Rectified(bgr_r, gray_r, float(working_px_per_mm), raw, g, residual, tiles)
