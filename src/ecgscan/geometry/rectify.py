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


def _homography(gray: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """D64 homography (with its fallbacks) as a 3x3 matrix and output size; identity if none applies."""
    from cardiolens.layer1_ingestion.digitize import dewarp_page

    dewarp_page(gray)
    hm, size = dewarp_page.last_transform  # type: ignore[attr-defined]
    return np.asarray(hm, dtype=np.float64), (int(size[0]), int(size[1]))


def _warp(img: np.ndarray, m: np.ndarray, size: tuple[int, int], border) -> np.ndarray:
    import cv2

    return cv2.warpPerspective(img, m, size, flags=cv2.INTER_LINEAR, borderValue=border)


def rectify(bgr: np.ndarray, *, working_px_per_mm: float, raw_grid: GridEstimate | None = None,
            max_upscale: float = 1.0, fit_px_per_mm: float | None = None) -> Rectified:
    """Area-downscale to the working scale once, then every step only *measures* on an
    intermediate image; the result is one warp by the composed matrix F * R * H (a 2-3 px
    trace stroke loses detail at every resampling)."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import estimate_skew

    raw = raw_grid or _fast_grid(bgr)
    if raw.px_per_mm is None:
        raise ValueError("no grid")
    s = min(max_upscale, working_px_per_mm / raw.px_per_mm)
    if s < working_px_per_mm / raw.px_per_mm:
        working_px_per_mm = raw.px_per_mm * s  # capped upscale: work at what the photo allows
    size = (int(round(bgr.shape[1] * s)), int(round(bgr.shape[0] * s)))
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Perspective (fitted at the working scale; fit_px_per_mm < working was measured to lose the traces).
    m_h, size = _homography(gray_s)
    gray_i = _warp(gray_s, m_h, size, 255)
    # Rotation: the texture homography steps aside for affine cases (D64), so de-skew here.
    skew = estimate_skew(gray_i)
    m_r = np.eye(3)
    if abs(skew) > 0.2:
        m_r = np.vstack([cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), skew, 1.0), [0.0, 0.0, 1.0]])
    bgr_i = _warp(small, m_r @ m_h, size, (255, 255, 255))
    if np.allclose(m_h, np.eye(3)) and tuple(size) == tuple(small.shape[1::-1]):
        # Scan / affine path: the area-downscale from the photo's own (precise, full-resolution)
        # pitch already gives the working scale; a "correcting" near-1 resample would only blur
        # the stroke (clean scans: QRS error 5 -> 18 ms) on a 0.4 % pitch read.
        g = GridEstimate(raw.px_per_mm_x * s, raw.px_per_mm_y * s, raw.score_x, raw.score_y, raw.channel,
                         raw.confidence)
        bgr_r = bgr_i
    else:
        g = estimate_grid(bgr_i)
        if g.px_per_mm_x is None or g.px_per_mm_y is None:
            g = GridEstimate(raw.px_per_mm * s, raw.px_per_mm * s, raw.score_x, raw.score_y, raw.channel,
                             raw.confidence * 0.5)
        # Perspective changed the scale: exact working scale on both axes, folded into the
        # same single warp of the area-downscaled photo.
        fx, fy = working_px_per_mm / g.px_per_mm_x, working_px_per_mm / g.px_per_mm_y
        m = np.diag([fx, fy, 1.0]) @ m_r @ m_h
        out_size = (int(round(size[0] * fx)), int(round(size[1] * fy)))
        bgr_r = _warp(small, m, out_size, (255, 255, 255))
    gray_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY)
    residual, tiles = tile_residual(bgr_r, working_px_per_mm)
    return Rectified(bgr_r, gray_r, float(working_px_per_mm), raw, g, residual, tiles)
