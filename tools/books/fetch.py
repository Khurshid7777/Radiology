"""D216: download the owner's ECG library from Google Drive into data/books/lib/ (git-ignored).

    python tools/books/fetch.py [--manifest tools/books/manifest.json] [--out data/books/lib] [--workers 4]

The manifest holds only Drive ids (and, once known, the titles); the books themselves are copyrighted
and never go into git. Files are shared "anyone with the link"; drive.usercontent.google.com serves the
original bytes (``confirm=t`` skips the virus-scan page of large files). Resumable: a file already on disk
is not fetched again. After a run the manifest is updated with each file's name, size and sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def _name(headers) -> str | None:
    cd = headers.get("Content-Disposition") or ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd)
    if m:
        return urllib.parse.unquote(m.group(1))
    m = re.search(r'filename="([^"]+)"', cd)
    if not m:
        return None
    try:   # http.client decodes the header as latin-1; Drive sends UTF-8 bytes
        return m.group(1).encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return m.group(1)


def _safe(name: str, max_bytes: int = 180) -> str:
    """A file-system-safe name of at most ``max_bytes`` UTF-8 bytes (ext4 allows 255 with the id prefix)."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip() or "file"
    stem, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 6:
        stem, ext = name, ""
    while len((stem + (f".{ext}" if ext else "")).encode("utf-8")) > max_bytes:
        stem = stem[:-1]
    return stem + (f".{ext}" if ext else "")


def fetch(item: dict, out: Path, tries: int = 4) -> dict:
    fid = item["id"]
    if item.get("duplicate_of"):                 # same bytes as another id (sha256): kept once
        return {**item, "status": "duplicate"}
    have = list(out.glob(f"{fid}__*"))
    if have:
        p = have[0]
        return {**item, "file": p.name, "size": p.stat().st_size, "status": "cached"}
    for i in range(tries):
        try:
            req = urllib.request.Request(URL.format(id=fid), headers={"User-Agent": "cardiolens-books/1"})
            with urllib.request.urlopen(req, timeout=300) as r:
                ctype = r.headers.get("Content-Type", "")
                name = _name(r.headers)
                if "text/html" in ctype and not name:
                    return {**item, "status": "html_no_file"}       # not shared, or a Drive page
                tmp = out / f".{fid}.part"
                h = hashlib.sha256()
                with tmp.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
            dest = out / f"{fid}__{_safe(name or item.get('title') or fid)}"
            tmp.rename(dest)
            return {**item, "title": item.get("title") or name, "file": dest.name, "size": dest.stat().st_size,
                    "sha256": h.hexdigest(), "status": "ok"}
        except Exception as e:  # network hiccup: retry with backoff, then report
            err = f"{type(e).__name__}: {str(e)[:100]}"
            time.sleep(2 ** i)
    return {**item, "status": "error", "error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "tools/books/manifest.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data/books/lib")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(a.workers) as ex:
        done = list(ex.map(lambda it: fetch(it, a.out), man["files"]))
    for d in done:
        print(d["status"], d["id"], d.get("file", d.get("error", "")), file=sys.stderr)
    man["files"] = [{k: v for k, v in d.items() if k != "status" or v not in ("ok", "cached", "duplicate")} for d in done]
    a.manifest.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for d in done if d["status"] in ("ok", "cached", "duplicate"))
    print(f"{ok}/{len(done)} files, {sum(d.get('size', 0) for d in done) / 1e6:.0f} MB in {a.out}")


if __name__ == "__main__":
    main()
