"""Image intake (TZ §5.1): the original stays untouched; the pipeline works on a copy.

``load_image`` hashes the bytes exactly as received, reads the EXIF orientation and
returns an upright BGR working copy. No enhancement of any kind is applied here
(TZ §10: no ML "document enhancement" on the original).
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImageInput:
    original_sha256: str
    original_bytes: int
    mime_type: str | None
    exif_orientation: int | None
    width_px: int
    height_px: int
    bgr: np.ndarray             # upright working copy (EXIF orientation applied)


def _mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None


def exif_orientation(data: bytes) -> int | None:
    """EXIF Orientation (1..8) of a JPEG, read without extra dependencies; None if absent."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker, size = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            t = data[i + 10:i + 2 + size]
            endian = "little" if t[:2] == b"II" else "big"
            ifd = int.from_bytes(t[4:8], endian)
            if ifd + 2 > len(t):
                return None
            for k in range(int.from_bytes(t[ifd:ifd + 2], endian)):
                e = ifd + 2 + 12 * k
                if e + 12 <= len(t) and int.from_bytes(t[e:e + 2], endian) == 0x0112:
                    v = int.from_bytes(t[e + 8:e + 10], endian)
                    return v if 1 <= v <= 8 else None
            return None
        if marker in (0xDA, 0xD9):
            break
        i += 2 + size
    return None


class ImageRefused(ValueError):
    """An image the pipeline will not decode; ``code`` is its reject reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the file header, without decoding the pixels; None if unknown."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:   # reads the header only
                return int(im.width), int(im.height)
    except Image.DecompressionBombError:
        return 1 << 20, 1 << 20      # over PIL's own ceiling: certainly too large
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def check_image(data: bytes, *, max_megapixels: float, min_side_px: int) -> None:
    """Refuse before decoding: a 1-MB PNG can declare 32000 x 32000 px and decode to 3 GB
    (D98); an empty or 1-px image crashed the frame checks."""
    if not data:
        raise ImageRefused("NOT_AN_ECG", "empty file")
    size = image_size(data)
    if size is None:
        return                       # not readable from the header: the decoder decides
    w, h = size
    if w * h > max_megapixels * 1e6:
        raise ImageRefused("IMAGE_TOO_LARGE", f"{w} x {h} px is over {max_megapixels:g} MP")
    if min(w, h) < min_side_px:
        raise ImageRefused("LOW_RESOLUTION", f"{w} x {h} px is below {min_side_px} px on a side")


def load_image(source: str | Path | bytes, *, max_megapixels: float = 60.0, min_side_px: int = 200) -> ImageInput:
    import cv2

    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    check_image(data, max_megapixels=max_megapixels, min_side_px=min_side_px)
    arr = np.frombuffer(data, dtype=np.uint8)
    # IMREAD_COLOR applies the EXIF orientation (OpenCV >= 3.1): the copy is upright.
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageRefused("NOT_AN_ECG", "not a decodable image")
    if min(bgr.shape[:2]) < min_side_px:
        raise ImageRefused("LOW_RESOLUTION", f"image is below {min_side_px} px on a side")
    return ImageInput(original_sha256=hashlib.sha256(data).hexdigest(), original_bytes=len(data),
                      mime_type=_mime(data), exif_orientation=exif_orientation(data),
                      width_px=int(bgr.shape[1]), height_px=int(bgr.shape[0]), bgr=bgr)
