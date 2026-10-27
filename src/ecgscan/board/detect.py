"""Find the board in a photo and map its zone to an exact-scale image."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecgscan.board.spec import BoardSpec


@dataclass
class BoardFit:
    spec: BoardSpec
    H: np.ndarray                    # photo px -> board mm
    marker_ids: list[int]
    residual_mm: float               # RMS of the marker corners off the homography
    min_px_per_mm: float             # the photo's resolution at the worst point of the zone
    mean_px_per_mm: float
    zone_in_frame: bool              # the whole zone was photographed
    ok: bool = True
    corners_only: bool = False       # SD-075: fitted to the 4 corner markers, the middle ones were off
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def _local_scale(H_inv: np.ndarray, spec: BoardSpec, n: int = 5) -> np.ndarray:
    """Photo px per board mm along x and y at an n x n lattice over the zone."""
    x0, y0, x1, y1 = spec.zone_mm
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    p = np.stack([xs.ravel(), ys.ravel()], 1)

    def proj(q: np.ndarray) -> np.ndarray:
        r = np.c_[q, np.ones(len(q))] @ H_inv.T
        return r[:, :2] / r[:, 2:]

    c = proj(p)
    dx = np.linalg.norm(proj(p + [1.0, 0.0]) - c, axis=1)
    dy = np.linalg.norm(proj(p + [0.0, 1.0]) - c, axis=1)
    return np.concatenate([dx, dy])


def _markers(gray: np.ndarray, dictionary: str, max_side: int) -> tuple[list[np.ndarray], list[int]]:
    import cv2

    h, w = gray.shape
    f = min(1.0, max_side / max(h, w))
    small = gray if f == 1.0 else cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(small)
    if ids is None:
        return [], []
    out = []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for c in corners:
        pts = (c.reshape(4, 2) / f).astype(np.float32)
        if f < 1.0:
            # Corners found on the reduced copy, refined on the photo itself.
            side = float(np.linalg.norm(pts[0] - pts[1]))
            win = int(max(2, min(10, side / 8)))
            pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (win, win), (-1, -1), crit).reshape(4, 2)
        out.append(pts.astype(np.float64))
    return out, [int(i) for i in ids.ravel()]


def _errors(H: np.ndarray, img_pts: np.ndarray, mm_pts: np.ndarray) -> np.ndarray:
    proj = np.c_[img_pts, np.ones(len(img_pts))] @ H.T
    return np.linalg.norm(proj[:, :2] / proj[:, 2:] - mm_pts, axis=1)


def _rms(H: np.ndarray, img_pts: np.ndarray, mm_pts: np.ndarray) -> float:
    return float(np.sqrt(np.mean(_errors(H, img_pts, mm_pts) ** 2)))


def detect_board(bgr: np.ndarray, specs: list[BoardSpec], *, min_markers: int = 3, max_residual_mm: float = 1.0,
                 max_side_px: int = 2000, whole_zone: bool = True) -> BoardFit | None:
    """The board seen in ``bgr``, or ``None`` when no board marker is in the photo.

    A fit with ``ok=False`` means a board is there but cannot be trusted (too few
    markers, corners off a plane): the caller falls back to the grid and says why.
    ``whole_zone=False``: a part of a set (SD-066), the zone may extend out of frame."""
    import cv2

    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found: dict[str, tuple[list[np.ndarray], list[int]]] = {}
    for dictionary in sorted({s.dictionary for s in specs}):
        found[dictionary] = _markers(gray, dictionary, max_side_px)
    best: tuple[BoardSpec, list[np.ndarray], list[int]] | None = None
    for s in specs:
        corners, ids = found[s.dictionary]
        mine = [(c, i) for c, i in zip(corners, ids, strict=True) if i in s.markers]
        # A marker id seen twice (a reflection, a second board): neither copy is used.
        counts = {i: sum(1 for _, j in mine if j == i) for _, i in mine}
        mine = [(c, i) for c, i in mine if counts[i] == 1]
        if mine and (best is None or len(mine) > len(best[2])):
            best = (s, [c for c, _ in mine], [i for _, i in mine])
    if best is None:
        return None
    spec, corners, ids = best
    img_pts = np.concatenate(corners)
    mm_pts = np.concatenate([spec.corners_mm(i) for i in ids])
    if len(ids) < min_markers:
        return BoardFit(spec, np.eye(3), sorted(ids), float("inf"), 0.0, 0.0, False, ok=False,
                        reason=f"{len(ids)} of {len(spec.markers)} markers seen (need {min_markers})")
    H, _ = cv2.findHomography(img_pts, mm_pts, cv2.RANSAC, max_residual_mm)
    if H is None:
        return BoardFit(spec, np.eye(3), sorted(ids), float("inf"), 0.0, 0.0, False, ok=False,
                        reason="no homography")
    residual = _rms(H, img_pts, mm_pts)
    note = None
    corner = spec.corner_ids
    if residual > max_residual_mm and set(corner) <= set(ids):
        # SD-075: a printer's uneven feed moves the middle markers by 2-3 mm and the corners
        # least. The plane is fitted to the corners alone; the paper grid then has to give
        # the scale and show the page flat (pipeline._zone_rect), else the board is not used.
        pick = np.concatenate([np.arange(4) + 4 * ids.index(i) for i in corner])
        Hc, _ = cv2.findHomography(img_pts[pick], mm_pts[pick], 0)
        if Hc is not None and _rms(Hc, img_pts[pick], mm_pts[pick]) <= max_residual_mm:
            off = _errors(Hc, img_pts, mm_pts).max()
            note = f"middle markers up to {off:.1f} mm off the corner fit (uneven print): plane from the corners"
            H, residual = Hc, _rms(Hc, img_pts[pick], mm_pts[pick])
    H_inv = np.linalg.inv(H)
    scales = _local_scale(H_inv, spec)
    x0, y0, x1, y1 = spec.zone_mm
    zc = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], np.float64) @ H_inv.T
    zc = zc[:, :2] / zc[:, 2:]
    h, w = gray.shape
    in_frame = bool((zc[:, 0] >= -1).all() and (zc[:, 0] <= w).all() and (zc[:, 1] >= -1).all() and (zc[:, 1] <= h).all())
    fit = BoardFit(spec, H, sorted(ids), residual, float(scales.min()), float(scales.mean()), in_frame,
                   corners_only=note is not None, notes=[note] if note else [])
    if residual > max_residual_mm:
        fit.ok, fit.reason = False, f"marker corners {residual:.2f} mm off a plane (board bent or a marker misread)"
    elif whole_zone and not in_frame:
        fit.ok, fit.reason = False, "the board's zone is not wholly in the photo"
    return fit


def _zone_transform(fit: BoardFit, px_per_mm: float, shape: tuple[int, int],
                    origin_mm: tuple[float, float] | None = None) -> tuple[np.ndarray, float]:
    """Photo px -> zone canvas px, and the factor the photo is first reduced by (never
    upscaled by more than the photo holds: a finer photo is area-averaged down first,
    a 3x reduction inside warpPerspective would alias the grid)."""
    f = min(1.0, 1.25 * px_per_mm / fit.mean_px_per_mm)
    H = fit.H
    if f < 1.0:
        h, w = shape
        H = H @ np.diag([w / round(w * f), h / round(h * f), 1.0])
    x0, y0 = origin_mm or fit.spec.zone_mm[:2]
    S = np.array([[px_per_mm, 0, -x0 * px_per_mm], [0, px_per_mm, -y0 * px_per_mm], [0, 0, 1]], np.float64)
    return S @ H, f


def zone_shape(spec: BoardSpec, px_per_mm: float) -> tuple[int, int]:
    zw, zh = spec.zone_size_mm
    return int(round(zh * px_per_mm)), int(round(zw * px_per_mm))


def warp_zone(bgr: np.ndarray, fit: BoardFit, px_per_mm: float, *, with_mask: bool = False,
              stretch: tuple[float, float] = (1.0, 1.0),
              area_mm: tuple[float, float, float, float] | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """The zone at exactly ``px_per_mm`` pixels per board millimetre; with ``with_mask``
    also where the photo covers it (bool). ``stretch``: x and y factors on top, in the same
    warp (SD-075: the paper grid's correction of the board scale, without a second resample).
    ``area_mm``: another board rectangle instead of the zone (SD-077: up to the markers)."""
    import cv2

    h, w = bgr.shape[:2]
    M, f = _zone_transform(fit, px_per_mm, (h, w), None if area_mm is None else area_mm[:2])
    src = bgr if f == 1.0 else cv2.resize(bgr, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    if area_mm is None:
        zh, zw = zone_shape(fit.spec, px_per_mm)
    else:
        zh, zw = int(round((area_mm[3] - area_mm[1]) * px_per_mm)), int(round((area_mm[2] - area_mm[0]) * px_per_mm))
    if stretch != (1.0, 1.0):
        M = np.diag([stretch[0], stretch[1], 1.0]) @ M
        zh, zw = int(round(zh * stretch[1])), int(round(zw * stretch[0]))
    out = cv2.warpPerspective(src, M, (zw, zh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if not with_mask:
        return out
    ones = np.full(src.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(ones, M, (zw, zh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    return out, mask > 0


def scale_map(fit: BoardFit, px_per_mm: float, n: int = 9) -> np.ndarray:
    """The photo's own px per mm (the worse axis) at every pixel of the zone canvas."""
    import cv2

    x0, y0, x1, y1 = fit.spec.zone_mm
    xs, ys = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    p = np.stack([xs.ravel(), ys.ravel()], 1)
    H_inv = np.linalg.inv(fit.H)

    def proj(q: np.ndarray) -> np.ndarray:
        r = np.c_[q, np.ones(len(q))] @ H_inv.T
        return r[:, :2] / r[:, 2:]

    c = proj(p)
    s = np.minimum(np.linalg.norm(proj(p + [1.0, 0.0]) - c, axis=1), np.linalg.norm(proj(p + [0.0, 1.0]) - c, axis=1))
    zh, zw = zone_shape(fit.spec, px_per_mm)
    return cv2.resize(s.reshape(n, n).astype(np.float32), (zw, zh), interpolation=cv2.INTER_LINEAR)
