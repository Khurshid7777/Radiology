"""Personal-data zones of a recorder's sheet mapped into a photo (S3, TZ 18.6; owner's decision 4,
SD-070): the app masks them before upload; the server uses the same mapping to check coverage.

A zone is ``[x0, y0, x1, y1]`` in fractions of the upright sheet (x along the paper, y down), from
the device profile's ``phi_zones``. The sheet is the quadrilateral the doctor confirmed on the
phone (``capture.corners.final``: top-left, top-right, bottom-right, bottom-left in photo pixels)
turned by ``capture.rotate_deg`` clockwise to stand upright (SD-071). A zone grows by ``margin``
(fractions of the sheet) on every side before mapping: a corner placed a little off still covers
the print. The Dart twin is ``mobile/ecg_scanner/lib/masking/zone_mask.dart``; the reference vectors
in ``tests/scanner/fixtures/mask_zones/vectors.json`` keep the two in agreement.
"""

from __future__ import annotations

import numpy as np


def sheet_corners(corners: list[list[float]], rotate_deg: int = 0) -> list[list[float]]:
    """The photo corners of the upright sheet's TL, TR, BR, BL: a quarter turn clockwise makes
    the photo's bottom-left the sheet's top-left."""
    k = (int(rotate_deg) // 90) % 4
    return [list(map(float, corners[(i - k) % 4])) for i in range(4)]


def homography(quad: list[list[float]]) -> np.ndarray:
    """3x3 map of the unit square (0,0) (1,0) (1,1) (0,1) onto ``quad``."""
    src = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    a, b = [], []
    for (x, y), (u, v) in zip(src, quad, strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(a, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def zone_polygon(zone: list[float], corners: list[list[float]], rotate_deg: int = 0,
                 margin: float = 0.0) -> list[list[float]]:
    """The zone's four corners in photo pixels (TL, TR, BR, BL of the zone on the sheet)."""
    x0, y0, x1, y1 = (max(0.0, zone[0] - margin), max(0.0, zone[1] - margin),
                      min(1.0, zone[2] + margin), min(1.0, zone[3] + margin))
    h = homography(sheet_corners(corners, rotate_deg))
    out = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        p = h @ np.array([x, y, 1.0])
        out.append([round(float(p[0] / p[2]), 2), round(float(p[1] / p[2]), 2)])
    return out


def mask_zones(bgr: np.ndarray, zones: list[list[float]], corners: list[list[float]], rotate_deg: int = 0,
               margin: float = 0.0) -> np.ndarray:
    """A copy of ``bgr`` with every zone painted white."""
    import cv2

    out = bgr.copy()
    for z in zones:
        poly = np.round(np.array(zone_polygon(z, corners, rotate_deg, margin))).astype(np.int32)
        cv2.fillPoly(out, [poly], (255, 255, 255))
    return out


def covered(box: tuple[int, int, int, int], mask: np.ndarray) -> float:
    """Share of a pixel box (x0, y0, x1, y1) inside ``mask`` (True = masked)."""
    x0, y0, x1, y1 = box
    sub = mask[max(0, y0):y1, max(0, x0):x1]
    return float(sub.mean()) if sub.size else 1.0
