"""D216: page-level text index of the owner's library and the summary table docs/knowledge/LIBRARY.md.

    python tools/books/index.py [--lib data/books/lib] [--text data/books/text]

PDF: one text file per page (pdftotext); a page with < 80 characters counts as scanned. docx/pptx/xlsx:
the text of the XML parts. Text stays in data/books/text/ (git-ignored); only the table goes to git.
Search afterwards: ``grep -ril "мобитц" data/books/text`` gives book id and page.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPICS = (("атлас|atlas|банк|150|проблемн|ekgshki|allam|medicom", "атлас / случаи"),
          ("аритм|тахи|ритм|жэс|впв|wpw|фибрил|arrhythm|tachy|rhythm|agonal", "ритм"),
          ("омi|omi|stemi|инфаркт|ишем|окклюз|ствол|смит|оим|тэла|такоцубо|brugada|бругад|t-wave|т_зубец|юнг",
           "ишемия / ST-T"),
          ("холтер|хм|holter|смад|нагруз|эргоспир|ndr", "холтер / нагрузка"),
          ("экс|кардиостим|ресинхр|оптимайзер", "стимуляция"),
          ("норм|алгоритм|чек|pocket|рек|нифонтов|родионов|корзун|qrs|qt|линейк|электрод|отведен", "основы / нормы"))


def topic(title: str) -> str:
    t = title.lower()
    for pat, name in TOPICS:
        if re.search(pat, t):
            return name
    return "прочее"


def pdf_pages(p: Path, out: Path) -> tuple[int, int]:
    info = subprocess.run(["pdfinfo", str(p)], check=False, capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", info)
    n = int(m.group(1)) if m else 0
    out.mkdir(parents=True, exist_ok=True)
    texty = 0
    for i in range(1, n + 1):
        dest = out / f"page-{i:04d}.txt"
        if not dest.exists():
            txt = subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), str(p), "-"],
                                 check=False, capture_output=True, text=True).stdout
            dest.write_text(txt, encoding="utf-8")
        texty += len(dest.read_text(encoding="utf-8").strip()) >= 80
    return n, texty


def office_text(p: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    try:
        z = zipfile.ZipFile(p)
    except zipfile.BadZipFile:
        return 0
    parts = sorted(n for n in z.namelist() if re.match(r"(word/document|ppt/slides/slide\d+|xl/sharedStrings)\.xml", n))
    total = 0
    for k, name in enumerate(parts, 1):
        txt = re.sub(r"<[^>]+>", " ", z.read(name).decode("utf-8", "ignore"))
        txt = re.sub(r"\s+", " ", txt).strip()
        (out / f"part-{k:04d}.txt").write_text(txt, encoding="utf-8")
        total += len(txt)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", type=Path, default=ROOT / "data/books/lib")
    ap.add_argument("--text", type=Path, default=ROOT / "data/books/text")
    ap.add_argument("--manifest", type=Path, default=ROOT / "tools/books/manifest.json")
    ap.add_argument("--md", type=Path, default=ROOT / "docs/knowledge/LIBRARY.md")
    a = ap.parse_args()
    man = {f["id"]: f for f in json.loads(a.manifest.read_text(encoding="utf-8"))["files"]}
    rows = []
    for p in sorted(a.lib.glob("*__*")):
        # a Drive id may itself contain "__": match the longest known id that prefixes the name
        fid = max((i for i in man if p.name.startswith(i + "__")), key=len, default=p.name.split("__", 1)[0])
        name = p.name[len(fid) + 2:]
        ext = p.suffix.lower()
        out = a.text / fid
        pages = texty = 0
        kind = ext.lstrip(".") or "?"
        if ext == ".pdf":
            pages, texty = pdf_pages(p, out)
            kind = "pdf: текст" if pages and texty / pages >= 0.6 else ("pdf: скан" if texty == 0 else "pdf: смешанный")
        elif ext in (".docx", ".pptx", ".xlsx"):
            kind = f"{ext[1:]}: текст" if office_text(p, out) > 200 else f"{ext[1:]}: мало текста"
        title = man.get(fid, {}).get("title") or name
        rows.append({"id": fid, "title": title, "kind": kind, "pages": pages, "text_pages": texty,
                     "mb": round(p.stat().st_size / 1e6, 1), "topic": topic(title)})
    rows.sort(key=lambda r: (r["topic"], r["title"].lower()))
    lines = ["# Библиотека владельца (D216)", "",
             ("Индекс файлов из Google Drive владельца. Сами файлы — только локально (`data/books/lib/`, загрузка "
             "`python tools/books/fetch.py`), текст по страницам — `data/books/text/<id>/` (`python tools/books/index.py`)."),
             "", (f"Файлов: {len(rows)}, страниц PDF: {sum(r['pages'] for r in rows)}, "
             f"из них с текстом: {sum(r['text_pages'] for r in rows)}."), "",
             "| Тема | Название | Тип | Стр. (с текстом) | МБ | Drive id |", "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["title"].replace("|", "/")
        pg = f"{r['pages']} ({r['text_pages']})" if r["pages"] else "—"
        lines.append(f"| {r['topic']} | {t} | {r['kind']} | {pg} | {r['mb']} | `{r['id']}` |")
    a.md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(rows), "files indexed ->", a.md)


if __name__ == "__main__":
    main()
