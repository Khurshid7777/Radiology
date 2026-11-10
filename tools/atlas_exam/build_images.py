"""D216: rebuild the atlas exam images from the owner's library (after tools/books/fetch.py).

    python tools/atlas_exam/build_images.py [--cases 'tools/atlas_exam/cases/cases_*.csv'] [--out data/books/exam/png]

The case lists (our short answers, pages, crop boxes) are in git; the images are not (copyrighted). Each
atlas is rendered the way its case list was made: pdftoppm at the atlas's dpi, then the crop box; EKGshki
pages are phone photos taken out as is (pdfimages). Hand edits made while building the lists (bank 053:
an overlapping ECG whitened; bank 057: an answer burned into the image) are not repeated — those two
images differ from the originals the baseline used.
"""

from __future__ import annotations

import argparse
import csv
import glob
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ATLAS = {"bank": ("1Tgbljk_N84AKcrdkRojGkrDWMlFkzmuN", 200), "yung": ("1PnALMYdS5cTl8aArPDkrf2d5nihixxPC", 200),
         "hampton": ("1Ih771uVc6CE1Okewc-PHtMJNm2BUcvi1", 200), "medicom2": ("175y0-TEuN-cvfmtveqqhO5CRhLvJQbJl", 200),
         "allam": ("1towzg1n_hhd9E8C5xvxiSkntfDyz47AM", 73), "ekgshki": ("1rRij0__T6eWzTC91CUzGbxZV2OpmDHSs", None)}


def page_image(pdf: Path, page: int, dpi: int | None, tmp: Path) -> Image.Image:
    if dpi is None:   # the embedded photo as is
        subprocess.run(["pdfimages", "-j", "-f", str(page), "-l", str(page), str(pdf), str(tmp / "p")], check=True)
        return Image.open(min(tmp.glob("p-*"))).convert("RGB")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-f", str(page), "-l", str(page), str(pdf), str(tmp / "p")],
                   check=True)
    return Image.open(min(tmp.glob("p*.png"))).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "tools/atlas_exam/cases/cases_*.csv"))
    ap.add_argument("--lib", type=Path, default=ROOT / "data/books/lib")
    ap.add_argument("--out", type=Path, default=ROOT / "data/books/exam/png")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(a.cases)):
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if not r.get("png") or r.get("layout") == "split" or not r.get("ecg_page"):
                continue
            dest = a.out / Path(r["png"]).name
            if dest.exists():
                continue
            fid, dpi = ATLAS[r["atlas"]]
            pdf = next(a.lib.glob(f"{fid}__*.pdf"))
            with tempfile.TemporaryDirectory() as td:
                im = page_image(pdf, int(r["ecg_page"]), dpi, Path(td))
            if r.get("crop"):
                im = im.crop(tuple(int(float(v)) for v in r["crop"].split(",")))
            im.save(dest)
            n += 1
    print(n, "images built in", a.out)


if __name__ == "__main__":
    main()
