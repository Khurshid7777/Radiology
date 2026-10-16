"""Photo/scan defects for synthetic pages, with the page -> photo transform (TZ §11).

Physical paper effects (crumple, creases, stains, faded print — the last is a
render option) are applied on the flat page; then the page is laid on a desk,
photographed in perspective and rotated, lit unevenly (shadow, vignette, glare, low
light), blurred, sensor noise is added, resolution reduced and JPEG-compressed.
Seeded and deterministic. ``info["homography"]`` maps flat-page pixels to photo
pixels exactly unless ``info["nonrigid"]`` (crumple) is set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DESK_BGR = (96, 112, 128)


@dataclass
class DegradeSpec:
    background: float = 0.0     # desk margin around the page, fraction of the page size
    perspective: float = 0.0    # corner displacement, fraction of the page size
    rotation_deg: float = 0.0   # max |rotation|; the sign and value are drawn per case
    crumple: float = 0.0        # smooth displacement amplitude, mm
    creases: int = 0
    stains: int = 0
    glare: float = 0.0          # 0..1 strength of a saturated specular spot
    shadow: float = 0.0         # 0..1 darkening of a linear shadow
    vignette: float = 0.0
    low_light: float = 0.0      # 0..1 global darkening (noise is separate)
    noise: float = 0.0          # sensor noise sigma, grey levels
    blur_px: float = 0.0        # Gaussian sigma, px
    scale: float = 1.0          # final resize factor
    jpeg_quality: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _smooth_field(rng, h: int, w: int, cell_px: float) -> np.ndarray:
    import cv2

    gh, gw = max(2, int(h / cell_px) + 2), max(2, int(w / cell_px) + 2)
    f = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)


def _crumple(img: np.ndarray, rng, amp_px: float, cell_px: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    dx = _smooth_field(rng, h, w, cell_px) * amp_px
    dy = _smooth_field(rng, h, w, cell_px) * amp_px
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Light falls on the folds: shade by the local slope of the displacement.
    slope = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
    gain = np.clip(1.0 + 0.6 * slope, 0.75, 1.1)[:, :, None]
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _creases(img: np.ndarray, rng, n: int, ppm: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    shade = np.zeros((h, w), dtype=np.float32)
    for _ in range(n):
        if rng.random() < 0.5:
            x = rng.uniform(0.15, 0.85) * w
            p0, p1 = (x + rng.uniform(-0.05, 0.05) * w, 0), (x + rng.uniform(-0.05, 0.05) * w, h)
        else:
            y = rng.uniform(0.15, 0.85) * h
            p0, p1 = (0, y + rng.uniform(-0.05, 0.05) * h), (w, y + rng.uniform(-0.05, 0.05) * h)
        cv2.line(shade, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1.0, max(1, int(0.6 * ppm)))
    shade = cv2.GaussianBlur(shade, (0, 0), max(1.0, 0.8 * ppm))
    shade /= max(1e-6, shade.max())
    return np.clip(img.astype(np.float32) * (1.0 - 0.25 * shade[:, :, None]), 0, 255).astype(np.uint8)


def _stains(img: np.ndarray, rng, n: int, ppm: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for _ in range(n):
        mask = np.zeros((h, w), dtype=np.float32)
        c = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h))
        ax = (int(rng.uniform(8, 25) * ppm), int(rng.uniform(6, 20) * ppm))
        cv2.ellipse(mask, c, ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, 2.0 * ppm))[:, :, None]
        tint = np.array([rng.uniform(0.55, 0.8), rng.uniform(0.7, 0.9), rng.uniform(0.85, 1.0)], dtype=np.float32)
        out *= 1.0 - mask * (1.0 - tint) * rng.uniform(0.6, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _project(img: np.ndarray, spec: DegradeSpec, rng) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = img.shape[:2]
    pad = int(spec.background * max(h, w))
    hm = np.eye(3)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=DESK_BGR)
        hm = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float64)
    H, W = img.shape[:2]
    if spec.perspective > 0 or spec.rotation_deg:
        src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * spec.perspective * np.array([W, H], np.float32)
        ang = np.deg2rad(rng.uniform(-1, 1) * spec.rotation_deg)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
        dst = (dst - [W / 2, H / 2]) @ rot.T + [W / 2, H / 2]
        dst -= dst.min(axis=0)
        p = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        out_w, out_h = int(np.ceil(dst[:, 0].max())) + 1, int(np.ceil(dst[:, 1].max())) + 1
        img = cv2.warpPerspective(img, p, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=DESK_BGR)
        hm = p.astype(np.float64) @ hm
    return img, hm


def _lighting(img: np.ndarray, spec: DegradeSpec, rng) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32, copy=False)
    if spec.shadow > 0:
        ang = rng.uniform(0, 2 * np.pi)
        proj = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
        proj = (proj - proj.min()) / max(1e-6, np.ptp(proj))
        f *= (1.0 - spec.shadow * np.clip(1.5 * proj - 0.5, 0, 1))[:, :, None]
    if spec.vignette > 0:
        rr = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        f *= (1.0 - spec.vignette * rr ** 2)[:, :, None]
    if spec.glare > 0:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        sig = rng.uniform(0.04, 0.09) * max(h, w)
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2)))
        f += (spec.glare * 255.0 * 1.6 * g)[:, :, None]
    if spec.low_light > 0:
        f *= 1.0 - spec.low_light
    return np.clip(f, 0, 255).astype(np.uint8)


def degrade(img: np.ndarray, spec: DegradeSpec, *, seed: int, px_per_mm: float) -> tuple[np.ndarray, dict]:
    """Apply ``spec`` to a rendered page; return (image, info)."""
    import cv2

    rng = np.random.default_rng(seed)
    out = img.copy()
    if spec.crumple > 0:
        out = _crumple(out, rng, spec.crumple * px_per_mm, cell_px=40.0 * px_per_mm)
    if spec.creases:
        out = _creases(out, rng, spec.creases, px_per_mm)
    if spec.stains:
        out = _stains(out, rng, spec.stains, px_per_mm)
    out, hm = _project(out, spec, rng)
    out = _lighting(out, spec, rng)
    if spec.blur_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), spec.blur_px)
    if spec.noise > 0:
        out = np.clip(out.astype(np.float32) + spec.noise * rng.standard_normal(out.shape, dtype=np.float32),
                      0, 255).astype(np.uint8)
    if spec.scale != 1.0:
        h, w = out.shape[:2]
        out = cv2.resize(out, (max(1, int(w * spec.scale)), max(1, int(h * spec.scale))), interpolation=cv2.INTER_AREA)
        hm = np.diag([spec.scale, spec.scale, 1.0]) @ hm
    if spec.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(spec.jpeg_quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    page_h, page_w = img.shape[:2]
    corners = np.array([[0, 0, 1], [page_w, 0, 1], [page_w, page_h, 1], [0, page_h, 1]], dtype=np.float64) @ hm.T
    info = {
        "homography": hm.round(6).tolist(),
        "page_quad_px": (corners[:, :2] / corners[:, 2:3]).round(1).tolist(),
        "nonrigid": spec.crumple > 0,
        # Linear scale of the page at its centre in the photo: true px/mm there = px_per_mm * scale.
        "centre_scale": _local_scale(hm, page_w / 2, page_h / 2),
    }
    return out, info


def _local_scale(hm: np.ndarray, u: float, v: float) -> float:
    """sqrt|det J| of the homography at (u, v)."""
    x, y, z = hm @ np.array([u, v, 1.0])
    j = (hm[:2, :2] * z - np.outer([x, y], hm[2, :2])) / z ** 2
    return round(float(np.sqrt(abs(np.linalg.det(j)))), 5)
