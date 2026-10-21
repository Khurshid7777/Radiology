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
    method: str = ""


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


def _homography(gray: np.ndarray, px_per_mm: float, texture_px_per_mm: float = 7.0
                ) -> tuple[np.ndarray, tuple[int, int], str]:
    """Perspective homography (3x3, output size, method).

    The D64 grid-texture fit (CardioLens) is run on a copy scaled to ~7 px/mm: it
    searches 4-9.5 px/mm grid pitches, so at the 10 px/mm working scale it always
    missed and the slow Hough-based fallbacks (quadrilateral, vanishing points,
    affine refine: 11+ s) ran on every page. An affine fit (rotation / scan) means no
    perspective: the de-skew step handles it. Only when the texture fit is impossible
    (too few grid tiles, poor residual) does the full legacy cascade run.
    """
    import cv2

    from cardiolens.layer1_ingestion import digitize as D

    h, w = gray.shape
    f = min(1.0, texture_px_per_mm / px_per_mm)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 0.999 else gray
    samples = D._local_grid_frequencies(small)
    if len(samples) >= D.GRID_TEXTURE_MIN_TILES:
        hm_s, p_s, rms = D._fit_grid_homography(samples)
        if rms > D.GRID_TEXTURE_MAX_RMS:
            # Outlier tiles (text, a trace-dominated tile, the 5-mm harmonic picked as the
            # peak) inflate the RMS although the robust fit is fine: drop them and refit.
            res = _tile_residuals(hm_s, p_s, samples)
            keep = [smp for smp, r in zip(samples, res, strict=True) if r <= 3.0 * float(np.median(res))]
            if len(keep) >= D.GRID_TEXTURE_MIN_TILES and len(keep) >= 0.6 * len(samples):
                hm_s, p_s, rms = D._fit_grid_homography(keep)
        if rms <= D.GRID_TEXTURE_MAX_RMS:
            sc = np.diag([f, f, 1.0])
            hm = np.linalg.inv(sc) @ hm_s @ sc                       # same map at full scale
            if abs(hm[2, 0]) * w + abs(hm[2, 1]) * h < D.QUAD_MIN_PERSPECTIVE:
                return np.eye(3), (w, h), "affine"
            hi = np.linalg.inv(hm)
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ hi.T
            corners = corners[:, :2] / corners[:, 2:3]
            mn, mx = corners.min(axis=0), corners.max(axis=0)
            out_w, out_h = int(round(mx[0] - mn[0])), int(round(mx[1] - mn[1]))
            if 50 < out_w < 4 * w and 50 < out_h < 4 * h:
                shift = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]])
                return shift @ hi, (out_w, out_h), "grid_texture"
    D.dewarp_page(gray)
    hm, size = D.dewarp_page.last_transform  # type: ignore[attr-defined]
    return np.asarray(hm, dtype=np.float64), (int(size[0]), int(size[1])), "legacy_cascade"


def _tile_residuals(hm: np.ndarray, p: float, samples) -> np.ndarray:
    """Per tile: how far its two grid frequency vectors are from one square grid under ``hm``
    (the residual of CardioLens ``_fit_grid_homography``, per tile, in units of 1/p)."""
    from cardiolens.layer1_ingestion.digitize import _homography_jacobian

    hi = np.linalg.inv(hm)
    e1, e2 = np.array([1.0 / p, 0.0]), np.array([0.0, 1.0 / p])
    out = []
    for x, y, fa, fb in samples:
        uvw = hi @ np.array([x, y, 1.0])
        jac = _homography_jacobian(hm, uvw[0] / uvw[2], uvw[1] / uvw[2])
        out.append(float(np.hypot(*((jac.T @ fa - e1) * p)) + np.hypot(*((jac.T @ fb - e2) * p))))
    return np.array(out)


def line_angles(gray: np.ndarray) -> tuple[float, float]:
    """Tilt (deg) of the long grid / trace lines: rows as dy/dx, columns as dx/dy.

    One Hough pass for both (CardioLens ``estimate_skew`` per direction was two passes,
    0.8 s each): the same Canny / Hough settings, rows at least a third of the width
    long, columns a third of the height; median of each, 0 when none is found.
    """
    import cv2

    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    # maxLineGap small: JPEG 8x8 block edges would chain into long axis-aligned "lines".
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=max(50, min(h, w) // 3),
                            maxLineGap=5)
    rows, cols = [], []
    for x1, y1, x2, y2 in (np.asarray(lines).reshape(-1, 4) if lines is not None else []):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = np.hypot(dx, dy)
        if length >= max(50, w // 3) and abs(a := np.degrees(np.arctan2(dy, dx))) <= 15:
            rows.append(a)
        if dy < 0:
            dx, dy = -dx, -dy
        if length >= max(50, h // 3) and abs(b := np.degrees(np.arctan2(dx, dy))) <= 15:
            cols.append(b)
    return (float(np.median(rows)) if rows else 0.0), (float(np.median(cols)) if cols else 0.0)


def _warp(img: np.ndarray, m: np.ndarray, size: tuple[int, int], border) -> np.ndarray:
    import cv2

    return cv2.warpPerspective(img, m, size, flags=cv2.INTER_LINEAR, borderValue=border)


def rectify(bgr: np.ndarray, *, working_px_per_mm: float, raw_grid: GridEstimate | None = None,
            max_upscale: float = 1.0, fit_px_per_mm: float | None = None,
            shear_min_deg: float | None = None) -> Rectified:
    """Area-downscale to the working scale once, then every step only *measures* on an
    intermediate image; the result is one warp by the composed matrix F * R * H (a 2-3 px
    trace stroke loses detail at every resampling)."""
    import cv2

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
    m_h, size, method = _homography(gray_s, working_px_per_mm)
    gray_i = _warp(gray_s, m_h, size, 255)
    # Rotation: the texture homography steps aside for affine cases (D64), so de-skew here.
    # After a texture homography the grid is already square (both angles read 0 at the
    # 0.25 deg Hough step): the 0.7-1.4 s Hough pass is skipped.
    skew, cols = line_angles(gray_i) if method != "grid_texture" else (0.0, 0.0)
    m_r = np.eye(3)
    if abs(skew) > 0.2:
        m_r = np.vstack([cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), skew, 1.0), [0.0, 0.0, 1.0]])
    if shear_min_deg is not None and abs(skew + cols) > shear_min_deg:
        # A photo the homography leaves to the affine path can still be sheared: rows and
        # columns of the grid tilted by different angles (rows 2.7 deg, columns -1.5 deg). One
        # rotation leaves 1.2 deg, ~20 ms of time offset per row of a 3x4 page (photo_clean-002,
        # SD-054). Map both measured directions back onto the axes, lengths kept.
        h, v = np.radians(skew), np.radians(cols)
        a = np.linalg.inv(np.array([[np.cos(h), np.sin(v)], [np.sin(h), np.cos(v)]]))
        c = np.array([size[0] / 2, size[1] / 2])
        m_r = np.eye(3)
        m_r[:2, :2], m_r[:2, 2] = a, c - a @ c
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
    return Rectified(bgr_r, gray_r, float(working_px_per_mm), raw, g, residual, tiles, method)
