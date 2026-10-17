"""Trace segmentation behind one interface (TZ §5.5).

``Segmenter.segment(bgr, px_per_mm)`` returns a ``TraceMask``: a boolean ink mask of
the trace plus a darkness map (0..1) used for sub-pixel centroids. Implementations:

* ``ClassicSegmenter`` — colour/brightness rules (default, no training);
* ``OnnxSegmenter`` — a learned model (M6). The interface is fixed now; without a
  model file it reports itself unavailable instead of pretending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class TraceMask:
    ink: np.ndarray            # bool [h, w]: trace pixels
    darkness: np.ndarray       # float32 [h, w] in 0..1: how far below the local paper level
    method: str


class Segmenter(Protocol):
    name: str

    def available(self) -> tuple[bool, str | None]: ...

    def segment(self, bgr: np.ndarray, px_per_mm: float) -> TraceMask: ...


class ClassicSegmenter:
    """Dark, neutral pixels relative to the local paper level.

    * the grid is coloured (chroma) and lighter than the trace — the chroma test and
      the relative-darkness test remove it without knowing its colour;
    * shadows, vignetting and low light are smooth — dividing by a local paper
      estimate (a wide max/median filter) cancels them (an Otsu threshold on raw
      brightness does not survive a shadow; Sauvola-style local normalisation does);
    * isolated specks (JPEG, noise, salt) are dropped by size.
    """

    name = "classic"

    def __init__(self, *, ratio: float = 0.62, max_chroma: int = 70, paper_window_mm: float = 4.0,
                 min_speck_px: int = 6):
        self.ratio = ratio
        self.max_chroma = max_chroma
        self.paper_window_mm = paper_window_mm
        self.min_speck_px = min_speck_px

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def segment(self, bgr: np.ndarray, px_per_mm: float) -> TraceMask:
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = max(5, int(round(self.paper_window_mm * px_per_mm)) | 1)
        # Paper level: a dilation (local max) ignores thin dark strokes, then a blur smooths it.
        paper = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        paper = cv2.blur(paper, (k, k))
        rel = gray / np.maximum(paper, 1.0)
        chroma = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
        ink = (rel < self.ratio) & (chroma < self.max_chroma)
        _n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < self.min_speck_px
        small[0] = False
        ink &= ~small[labels]
        ink &= paper_mask(bgr, px_per_mm)
        darkness = np.clip((1.0 - rel) / (1.0 - self.ratio * 0.5), 0.0, 1.0).astype(np.float32)
        return TraceMask(ink, darkness, self.name)


def paper_mask(bgr: np.ndarray, px_per_mm: float, *, window_mm: float = 6.0, min_ratio: float = 1.15,
               close_mm: float = 8.0) -> np.ndarray:
    """Where the paper is: on the 1-mm grid the brightness differs much more between pixels
    half a period apart than one period apart; on a desk, a hand or the warp border the two
    are equal (bench: desk 0.5-0.9, paper 1.3-2.8, faded paper 1.5). Without this mask the
    dark neutral desk is "ink" and a desk band along the frame swamped the row projection.
    Holes (traces, text, glare) are closed; regions >= 20 % of the largest are kept."""
    import cv2

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = max(2, int(round(px_per_mm)))
    k = max(5, int(round(window_mm * px_per_mm)) | 1)
    ratio = None
    for ax in (0, 1):
        half = np.abs(g - np.roll(g, p // 2, axis=ax))
        full = np.abs(g - np.roll(g, p, axis=ax))
        r = cv2.blur(half, (k, k)) / (cv2.blur(full, (k, k)) + 1.0)
        ratio = r if ratio is None else np.maximum(ratio, r)
    paper = (ratio > min_ratio).astype(np.uint8)
    kc = max(3, int(round(close_mm * px_per_mm)) | 1)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    if n <= 1:
        return np.ones(paper.shape, dtype=bool)        # no grid texture found: mask nothing
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= 0.2 * areas.max()
    return keep[labels]


class OnnxSegmenter:
    """Learned segmentation (M6). Without a model it is unavailable, never a silent fallback."""

    name = "onnx"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def available(self) -> tuple[bool, str | None]:
        from pathlib import Path

        if not self.model_path or not Path(self.model_path).exists():
            return False, "TODO(M6): no ONNX segmentation model configured"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime not installed"
        return True, None

    def segment(self, bgr: np.ndarray, px_per_mm: float) -> TraceMask:  # pragma: no cover - M6
        _ok, why = self.available()
        raise NotImplementedError(why or "TODO(M6): ONNX segmentation inference")


def make_segmenter(cfg: dict) -> Segmenter:
    kind = cfg.get("kind", "classic")
    if kind == "onnx":
        seg = OnnxSegmenter(cfg.get("onnx_model"))
        ok, why = seg.available()
        if not ok:
            raise RuntimeError(f"segmenter 'onnx' unavailable: {why}")
        return seg
    c = cfg.get("classic", {})
    return ClassicSegmenter(**c)


def thin(mask: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """Zhang-Suen thinning to a 1-px skeleton (vectorised; the CardioLens D47 routine).

    A printed stroke is 2-4 px thick, so a steep QRS edge covers 3-4 columns and the
    column before the onset already looks deflected: QRS read ~10 ms wide, PR short,
    QT long (M3 measurement). On the skeleton the edge is back in its own column.
    """
    img = mask.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c1, c2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                c1, c2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            rm = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)
