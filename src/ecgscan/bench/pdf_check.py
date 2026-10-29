"""Does a report PDF open everywhere and show every flag (S4, TZ 18.14)?

Three independent engines stand in for the doctors' devices:

* **PDFium** (``pypdfium2``): Chrome's and Android's viewer;
* **Poppler** (``pdftoppm``, ``pdftotext``, ``pdffonts``): most Linux viewers;
* **MuPDF** (``pymupdf``): many mobile readers.

Each one renders every page (a blank page is a failure) and extracts the text; every expected
phrase must be found by every engine. Every font text is set in must be embedded: a PDF that relies on the
viewer's fonts shows Cyrillic and Uzbek ‘ as boxes on a phone without them. Apple's PDFKit (iOS,
macOS Preview) cannot run here.

    python -m ecgscan.bench.pdf_check report.pdf --expect "Скорость ленты" --out pages/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("­", "")).strip()


def _found(text: str, expect: list[str]) -> list[str]:
    flat = _norm(text)
    return [e for e in expect if _norm(e) not in flat]


def _ink(rgb: np.ndarray) -> float:
    """Fraction of non-white pixels: a page that rendered as nothing has ~0."""
    return float((rgb.min(axis=-1) < 200).mean())


def check_pdfium(path: Path, expect: list[str], out: Path | None) -> dict:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    text, ink = [], []
    for i, page in enumerate(doc):
        img = page.render(scale=1.5).to_pil().convert("RGB")
        ink.append(_ink(np.asarray(img)))
        if out:
            img.save(out / f"pdfium-{i + 1}.png")
        text.append(page.get_textpage().get_text_range())
    return {"pages": len(doc), "ink": ink, "missing": _found("\n".join(text), expect)}


def check_mupdf(path: Path, expect: list[str], out: Path | None) -> dict:
    import pymupdf

    doc = pymupdf.open(str(path))
    text, ink = [], []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=108)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        ink.append(_ink(rgb))
        if out:
            pix.save(str(out / f"mupdf-{i + 1}.png"))
        text.append(page.get_text())
    return {"pages": doc.page_count, "ink": ink, "missing": _found("\n".join(text), expect)}


def check_poppler(path: Path, expect: list[str], out: Path | None) -> dict:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "108", "-png", str(path), f"{tmp}/p"], check=True, capture_output=True)
        pages = sorted(Path(tmp).glob("p-*.png"))
        ink = [_ink(np.asarray(Image.open(p).convert("RGB"))) for p in pages]
        if out:
            for i, p in enumerate(pages):
                shutil.copyfile(p, out / f"poppler-{i + 1}.png")
    text = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], check=True, capture_output=True).stdout
    return {"pages": len(pages), "ink": ink, "missing": _found(text.decode("utf-8"), expect)}


def fonts(path: Path) -> list[dict]:
    """Every font of the PDF (``pdffonts``): name, embedded, subset, unicode map, and whether any
    text is set in it (MuPDF's spans). reportlab lists Helvetica on every page whether or not a
    character uses it: only the fonts text is set in must be embedded."""
    import pymupdf

    used = set()
    for page in pymupdf.open(str(path)):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                used.update(span["font"] for span in line["spans"])
    lines = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
    rows = []
    for ln in lines[2:]:
        parts = ln.split()
        # name type [type…] encoding emb sub uni object ID: the last five columns are fixed
        emb, sub, uni = parts[-5:-2]
        name = parts[0]
        rows.append({"name": name, "embedded": emb == "yes", "subset": sub == "yes", "unicode": uni == "yes",
                     "used": name.split("+")[-1] in used})
    return rows


ENGINES = {"pdfium": check_pdfium, "poppler": check_poppler, "mupdf": check_mupdf}
MIN_INK = 0.002          # a page with less than 0.2 % ink rendered as nothing


def check(path: str | Path, expect: list[str], out_dir: str | Path | None = None) -> dict:
    path = Path(path)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    res = {"file": path.name, "bytes": path.stat().st_size, "engines": {}, "fonts": fonts(path)}
    for name, fn in ENGINES.items():
        try:
            res["engines"][name] = fn(path, expect, out)
        except Exception as e:                    # an engine that cannot open the file is the finding
            res["engines"][name] = {"error": f"{type(e).__name__}: {e}"}
    eng = res["engines"].values()
    res["ok"] = bool(
        all("error" not in e and not e["missing"] and min(e["ink"], default=0) >= MIN_INK for e in eng)
        and len({e.get("pages") for e in eng}) == 1
        and any(f["used"] for f in res["fonts"])
        and all(f["embedded"] and f["unicode"] for f in res["fonts"] if f["used"]))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--expect", action="append", default=[], help="a phrase every engine must find")
    ap.add_argument("--out", help="directory for the rendered pages")
    a = ap.parse_args(argv)
    res = check(a.pdf, a.expect, a.out)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
