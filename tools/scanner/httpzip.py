"""R8 (SD-111): read a few members of a huge zip over HTTP range requests (the PM-ECG-ID archive is 34 GB,
the disk allowance is smaller). Only bytes are fetched and stored; nothing from the archive is executed.

    python -I tools/scanner/httpzip.py URL --list
    python -I tools/scanner/httpzip.py URL --extract OUT_DIR NAME [NAME ...]
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    """A read-only seekable file over HTTP Range requests, with a small read-ahead cache."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self._cache = url, block, 0, (0, b"")
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except OSError:
                if attempt == 3:
                    raise
        return b""

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        c0, cb = self._cache
        if not (c0 <= self.pos and self.pos + n <= c0 + len(cb)):
            span = max(n, self.block)
            cb = self._fetch(self.pos, min(self.size, self.pos + span))
            self._cache = (self.pos, cb)
            c0 = self.pos
        out = self._cache[1][self.pos - c0:self.pos - c0 + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract", nargs="+")
    ap.add_argument("--names-file", help="with --extract OUT_DIR: member names, one per line")
    a = ap.parse_args()
    zf = zipfile.ZipFile(HttpFile(a.url))
    if a.list:
        for i in zf.infolist():
            print(f"{i.file_size}\t{i.filename}")
        return
    out = Path(a.extract[0])
    names = a.extract[1:] + (Path(a.names_file).read_text().split("\n") if a.names_file else [])
    for name in filter(None, names):
        if (out / name).exists():
            continue                                      # resumable
        target = (out / name).resolve()
        if out.resolve() not in target.parents:       # never write outside the target folder
            raise SystemExit(f"unsafe member name: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(name))


if __name__ == "__main__":
    main()
