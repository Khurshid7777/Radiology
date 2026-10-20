"""Debug and confirmation artifacts (TZ §5.9, §6.3).

* ``rectified.png`` — the page after rectification (working px/mm);
* ``mask.png`` — the trace segmentation;
* ``overlay.png`` — every extracted trace drawn over the rectified page, panel
  boxes and lead names, coloured by quality (green ok / amber doubtful / red bad):
  what the doctor checks on the confirmation screen;
* ``overlay_preview.jpg`` — the same, at most ``preview_width`` px wide (the full
  PNG of a 3000-px page is ~8 MB; a phone screen and the bot need ~0.3 MB).

Artifacts are derived from the working copy only: never the unmasked original. The
header and footer bands (name, ID, date) are blanked on the server whatever the client
did (D98, ``artifacts.mask_bands``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

QUALITY_COLOURS = ((0.8, (40, 160, 40)), (0.5, (0, 160, 230)), (0.0, (40, 40, 220)))   # BGR


def _colour(q: float) -> tuple[int, int, int]:
    for thr, c in QUALITY_COLOURS:
        if q >= thr:
            return c
    return QUALITY_COLOURS[-1][1]


def render_overlay(bgr: np.ndarray, readings) -> np.ndarray:
    import cv2

    out = bgr.copy()
    for rd in readings:
        p = rd.panel
        col = _colour(rd.quality)
        y = rd.y_px
        xs = np.arange(p.x0, p.x0 + len(y))
        ok = np.isfinite(y)
        # Draw continuous runs of valid samples as polylines; gaps stay visibly open.
        idx = np.flatnonzero(ok)
        if idx.size:
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            for seg in np.split(idx, splits):
                if seg.size >= 2:
                    pts = np.stack([xs[seg], np.round(y[seg])], axis=1).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out, [pts], False, col, 2, cv2.LINE_AA)
        if rd.trace.box_px:
            x0, _, x1, _ = rd.trace.box_px
            cv2.rectangle(out, (x0, int(p.baseline) - 2), (x1, int(p.baseline) + 2), col, 1)
        cv2.putText(out, f"{p.lead} {rd.quality:.2f}", (p.x0 + 4, int(p.baseline) - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    return out


def blank_outside(img: np.ndarray, keep_rows: tuple[int, int] | None, fill=255) -> np.ndarray:
    """``img`` with every row outside [top, bottom) set to ``fill`` (the page header/footer)."""
    if keep_rows is None:
        return img
    top, bottom = keep_rows
    out = img.copy()
    out[:max(0, top)] = fill
    out[max(0, bottom):] = fill
    return out


def write_artifacts(out_dir: str | Path, *, rectified: np.ndarray, mask: np.ndarray | None, readings,
                    preview_width: int = 1600, preview_quality: int = 85,
                    keep_rows: tuple[int, int] | None = None) -> dict[str, str]:
    """``keep_rows``: (top, bottom) of the trace rows; everything above and below — the header
    and footer with the patient's name, ID and date — is blanked in every image (D98)."""
    import cv2

    rectified = blank_outside(rectified, keep_rows)
    if mask is not None:
        mask = blank_outside(mask, keep_rows, fill=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"rectified_image": out_dir / "rectified.png", "trace_overlay": out_dir / "overlay.png",
             "trace_overlay_preview": out_dir / "overlay_preview.jpg"}
    cv2.imwrite(str(paths["rectified_image"]), rectified)
    overlay = render_overlay(rectified, readings)
    cv2.imwrite(str(paths["trace_overlay"]), overlay)
    f = min(1.0, preview_width / overlay.shape[1])
    small = cv2.resize(overlay, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1.0 else overlay
    cv2.imwrite(str(paths["trace_overlay_preview"]), small, [cv2.IMWRITE_JPEG_QUALITY, preview_quality])
    if mask is not None:
        paths["trace_mask"] = out_dir / "mask.png"
        cv2.imwrite(str(paths["trace_mask"]), (mask.astype(np.uint8) * 255))
    return {k: str(v) for k, v in paths.items()}
