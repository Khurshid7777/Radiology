"""Printable board: markers, the zone outline and millimetre rulers along it.

Printed at 100 % ("actual size", no "fit to page"): the scan does not depend on the
print scale being exact (the markers are measured where they are printed, the
grid cross-check catches a scaled print), but rulers let a user verify it.
"""

from __future__ import annotations

import numpy as np

from ecgscan.board.spec import BoardSpec


def render_board(spec: BoardSpec, px_per_mm: float = 300 / 25.4) -> np.ndarray:
    """Greyscale image of the board at ``px_per_mm`` (default 300 dpi)."""
    import cv2

    def p(v: float) -> int:
        return int(round(v * px_per_mm))

    w, h = p(spec.size_mm[0]), p(spec.size_mm[1])
    img = np.full((h, w), 255, np.uint8)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    side = p(spec.marker_mm)
    for mid, (cx, cy) in spec.markers.items():
        m = cv2.aruco.generateImageMarker(dic, mid, side)
        x0, y0 = p(cx) - side // 2, p(cy) - side // 2
        img[y0:y0 + side, x0:x0 + side] = m
    x0, y0, x1, y1 = spec.zone_mm
    line = max(1, p(0.3))
    cv2.rectangle(img, (p(x0), p(y0)), (p(x1), p(y1)), 150, line)
    # Rulers outside the zone (the scan never sees them): 1 mm ticks, 5 and 10 mm longer.
    for mm in range(int(x1 - x0) + 1):
        ln = 6 if mm % 10 == 0 else 4 if mm % 5 == 0 else 2
        x = p(x0 + mm)
        cv2.line(img, (x, p(y0) - p(1)), (x, p(y0 - 1 - ln)), 120, line)
        cv2.line(img, (x, p(y1) + p(1)), (x, p(y1 + 1 + ln)), 120, line)
    for mm in range(int(y1 - y0) + 1):
        ln = 6 if mm % 10 == 0 else 4 if mm % 5 == 0 else 2
        y = p(y0 + mm)
        cv2.line(img, (p(x0) - p(1), y), (p(x0 - 1 - ln), y), 120, line)
        cv2.line(img, (p(x1) + p(1), y), (p(x1 + 1 + ln), y), 120, line)
    scale = px_per_mm / 12
    cv2.putText(img, f"{spec.title}  |  ECG inside the frame, flat, 100% print", (p(x0) + p(5), p(y0) - p(12)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 90, max(1, int(round(2 * scale))), cv2.LINE_AA)
    return img


def render_sheets(spec: BoardSpec, px_per_mm: float = 300 / 25.4) -> list[np.ndarray]:
    """The board as the sheets it is printed on (SD-076); one image for a one-sheet board.

    Sheet 1 carries a dashed cut line at ``cut_mm``: cut along it and lay it over sheet 2 so
    its edge meets the zone outline and rulers printed there."""
    import cv2

    full = render_board(spec, px_per_mm)
    if not spec.sheets:
        return [full]

    def p(v: float) -> int:
        return int(round(v * px_per_mm))

    out = []
    n = len(spec.sheets)
    scale = px_per_mm / 12
    for k, (a, b) in enumerate(spec.sheets):
        tile = np.ascontiguousarray(full[:, p(a):p(b)])
        cut = spec.cut_mm
        if cut is not None and a < cut < b:
            x = p(cut - a)
            on = k == 0                                  # cut this one; the next shows where it goes
            for y in range(0, tile.shape[0], p(6)):
                cv2.line(tile, (x, y), (x, min(tile.shape[0], y + p(3 if on else 6))), 60 if on else 170,
                         max(1, p(0.3)))
            text = "cut along the dashed line, lay over sheet 2" if on else "sheet 1 edge here"
            cv2.putText(tile, text, (x - p(120) if on else x + p(2), tile.shape[0] - p(12)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, 90, max(1, int(round(2 * scale))), cv2.LINE_AA)
        cv2.putText(tile, f"{k + 1} / {n}", (p(8) if k else tile.shape[1] - p(30), p(12)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 90, max(1, int(round(2 * scale))), cv2.LINE_AA)
        out.append(tile)
    return out
