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

# Chosen on the D219 bench (even half; accepted on the odd half): x2 Lanczos, no smoothing, for pictures
# whose long side is under 2 000 px. "enabled": False restores the earlier behaviour exactly.
PREP = {"enabled": True, "max_long_px": 2000, "factor": 2.0, "method": "lanczos", "blur": 0.0}
# Provenance note on a record read from an enlarged picture: its QT is approximate (10-70 ms on the bench).
ENLARGED_NOTE = "picture enlarged before reading (D219): intervals approximate"
# Above the QTc limit by less than this, a long QT from an enlarged picture is not a critical alert.
ENLARGED_QTC_MARGIN_MS = 40.0


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


REDUCED_LONG_PX = 1280        # D275: the size of a messenger copy, which the scanner's coarse mode reads
REDUCE_FROM_PX = 1500         # only a picture longer than this is reduced


@contextmanager
def reduced(path: str | Path):
    """D275: a reduced temporary copy (long side 1280 px) of a large picture, or None. A small photo stretched in an
    editor to 3000 x 4000 carries no more detail, and its fine grid is blurred: the scanner's coarse mode (only for
    images up to 2000 px) reads the reduced copy where it refuses the stretched one."""
    from PIL import Image

    try:
        im = Image.open(path)
        im.load()
    except Exception:
        yield None
        return
    if max(im.size) <= REDUCE_FROM_PX:
        yield None
        return
    k = REDUCED_LONG_PX / max(im.size)
    small = im.convert("RGB").resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / (Path(path).stem + "_reduced.png")
        small.save(out)
        yield out
