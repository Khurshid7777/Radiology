"""Trace segmentation behind one interface (TZ §5.5).

``Segmenter.segment(bgr, px_per_mm)`` returns a ``TraceMask``: a boolean ink mask of
the trace plus a darkness map (0..1) used for sub-pixel centroids. Implementations:

* ``ClassicSegmenter`` — colour/brightness rules (default, no training);
* ``OnnxSegmenter`` — a learned model (M6). The interface is fixed now; without a
  model file it reports itself unavailable instead of pretending.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    Regions >= 20 % of the largest are kept and holes in them (traces, labels, glare) filled."""
    import cv2

    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = max(2, int(round(px_per_mm)))
    k = max(5, int(round(window_mm * px_per_mm)) | 1)
    ratio = None
    for ax in (0, 1):
        half = cv2.absdiff(g, np.roll(g, p // 2, axis=ax))       # = |g - roll|, without temporaries
        full = cv2.absdiff(g, np.roll(g, p, axis=ax))
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
    # Large lead labels have no grid texture either and left holes the closing did not
    # reach (the first 9 mm of a 6x2 row were cut out). Anything enclosed by paper is paper.
    from scipy.ndimage import binary_fill_holes

    return binary_fill_holes(keep[labels])


class OnnxSegmenter:
    """Learned stroke segmentation (M6, SD-052): a small U-Net run with onnxruntime.

    The model was trained on rectified pages at the working scale (synthetic only,
    ``ecgscan.segmentation.dataset``); its output probability is the darkness map
    (sub-pixel centroids) and ``ink`` is where it exceeds ``threshold``. Pages are run
    in overlapping tiles (memory stays bounded on a 3000-px page). The paper mask is
    applied as in the classic segmenter: the model has never seen a desk. Without a
    model file it is unavailable — never a silent fallback to the classic one.
    """

    name = "onnx"

    def __init__(self, model_path: str | None = None, *, threshold: float = 0.5, tile: int = 1024,
                 overlap: int = 32, sha256: str | None = None, threads: int | None = None):
        self.model_path = model_path
        self.threshold = threshold
        self.tile = tile
        self.overlap = overlap
        self.sha256 = sha256
        self.threads = threads
        self._session = None

    def _path(self):
        from pathlib import Path

        from ecgscan.config import repo_root

        if not self.model_path:
            return None
        p = Path(self.model_path)
        return p if p.is_absolute() else repo_root() / p

    def available(self) -> tuple[bool, str | None]:
        p = self._path()
        if p is None or not p.exists():
            return False, f"no ONNX segmentation model at {self.model_path!r}"
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime not installed"
        if self.sha256:
            import hashlib

            if hashlib.sha256(p.read_bytes()).hexdigest() != self.sha256:
                return False, "ONNX model does not match its sha256 (model_card.json)"
        return True, None

    def _sess(self):
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if self.threads:
                opts.intra_op_num_threads = self.threads
            self._session = ort.InferenceSession(str(self._path()), opts, providers=["CPUExecutionProvider"])
        return self._session

    def probability(self, bgr: np.ndarray) -> np.ndarray:
        """Stroke probability (float32, H×W) over the whole page, tile by tile."""
        sess = self._sess()
        h, w = bgr.shape[:2]
        out = np.zeros((h, w), np.float32)
        step = self.tile - 2 * self.overlap
        for y in range(0, h, step):
            for x in range(0, w, step):
                y0, x0 = max(0, y - self.overlap), max(0, x - self.overlap)
                y1, x1 = min(h, y + step + self.overlap), min(w, x + step + self.overlap)
                patch = bgr[y0:y1, x0:x1].astype(np.float32) / 255.0 - 0.5
                ph, pw = patch.shape[:2]
                padded = np.pad(patch, ((0, -ph % 8), (0, -pw % 8), (0, 0)), mode="edge")
                logits = sess.run(None, {"image": padded.transpose(2, 0, 1)[None]})[0][0, 0, :ph, :pw]
                prob = 1.0 / (1.0 + np.exp(-logits))
                # Keep the tile's centre only: the borders see half a context.
                cy0, cx0 = y - y0, x - x0
                cy1, cx1 = min(ph, cy0 + step), min(pw, cx0 + step)
                out[y:y + cy1 - cy0, x:x + cx1 - cx0] = prob[cy0:cy1, cx0:cx1]
        return out

    def segment(self, bgr: np.ndarray, px_per_mm: float) -> TraceMask:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"segmenter 'onnx' unavailable: {why}")
        # The paper mask (cv2, ~0.5 s) is computed while onnxruntime runs: both release the
        # GIL, the result is the same (SD-069).
        paper = _MASK_POOL.submit(paper_mask, bgr, px_per_mm)
        prob = self.probability(bgr)
        ink = (prob > self.threshold) & paper.result()
        return TraceMask(ink, prob, self.name)


_ONNX_CACHE: dict[tuple, OnnxSegmenter] = {}
_MASK_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ecgscan-mask")


def make_segmenter(cfg: dict) -> Segmenter:
    kind = cfg.get("kind", "classic")
    if kind == "onnx":
        o = cfg.get("onnx", {})
        key = (cfg.get("onnx_model"), o.get("threshold", 0.5), o.get("tile", 1024), o.get("overlap", 32),
               o.get("sha256"), o.get("threads"))
        seg = _ONNX_CACHE.get(key)
        if seg is None:                       # one onnxruntime session per model, not per scan
            seg = OnnxSegmenter(key[0], threshold=key[1], tile=key[2], overlap=key[3], sha256=key[4], threads=key[5])
            ok, why = seg.available()
            if not ok:
                raise RuntimeError(f"segmenter 'onnx' unavailable: {why}")
            _ONNX_CACHE[key] = seg
        return seg
    c = cfg.get("classic", {})
    return ClassicSegmenter(**c)


def thin(mask: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """Zhang-Suen thinning to a 1-px skeleton (vectorised; the CardioLens D47 routine).

    A printed stroke is 2-4 px thick, so a steep QRS edge covers 3-4 columns and the
    column before the onset already looks deflected: QRS read ~10 ms wide, PR short,
    QT long (M3 measurement). On the skeleton the edge is back in its own column.
    """
    # Computed on the stroke pixels only (~2 % of a page) instead of the whole frame at
    # every sub-step: the same rule, bit-identical skeleton, 4-6x faster (SD-063).
    img = np.pad(mask.astype(np.uint8), 1)
    w = img.shape[1]
    flat = img.ravel()
    offs = np.array([-w, -w + 1, 1, w + 1, w, w - 1, -1, -w - 1])     # P2..P9: N, NE, E, SE, S, SW, W, NW
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            idx = np.flatnonzero(flat)
            nb = flat[idx[:, None] + offs[None, :]]
            b = nb.sum(axis=1)
            seq = np.concatenate([nb, nb[:, :1]], axis=1)
            a = ((seq[:, :-1] == 0) & (seq[:, 1:] == 1)).sum(axis=1)
            p2, p4, p6, p8 = nb[:, 0], nb[:, 2], nb[:, 4], nb[:, 6]
            if step == 0:
                c1, c2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                c1, c2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            rm = (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if rm.any():
                flat[idx[rm]] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)
