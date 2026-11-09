"""D219: preparing a small ECG picture before it is read.

Atlas pages and messenger-compressed photos carry a whole 12-lead sheet in 1 000-1 600 px (4-6 px per mm
of grid), below what the readers and the scanner are built for. A picture whose long side is under
``max_long_px`` is enlarged by ``factor`` (and optionally smoothed) into a temporary PNG that is read in
its place. No information is added; the readers simply see the grid and the traces at the size they
expect. The parameters were chosen by tools/lowres_bench on PTB-XL renders with a known signal
(score = read x accurate - 3 x silent-wrong), see DECISIONS D219.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

# Set from the D219 bench; "enabled": False restores the earlier behaviour exactly.
PREP = {"enabled": False, "max_long_px": 2000, "factor": 2.0, "method": "lanczos", "blur": 0.0}


def needs_prep(size: tuple[int, int], prep: dict | None = None) -> bool:
    p = prep or PREP
    return bool(p["enabled"]) and max(size) < int(p["max_long_px"]) and float(p["factor"]) > 1.0


@contextmanager
def prepared(path: str | Path, prep: dict | None = None):
    """Yield the path to read: the original, or an enlarged temporary copy of a small picture."""
    from PIL import Image, ImageFilter

    p = prep or PREP
    try:
        im = Image.open(path)
        im.load()
    except Exception:  # not a picture PIL reads (a PDF, a DICOM): the readers decide
        yield path
        return
    if not needs_prep(im.size, p):
        yield path
        return
    method = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}.get(str(p["method"]), Image.LANCZOS)
    f = float(p["factor"])
    big = im.convert("RGB").resize((round(im.width * f), round(im.height * f)), method)
    if float(p.get("blur") or 0) > 0:
        big = big.filter(ImageFilter.GaussianBlur(float(p["blur"])))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / (Path(path).stem + "_prep.png")
        big.save(out)
        yield out
