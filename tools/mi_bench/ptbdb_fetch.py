"""D332: fetch PTBDB from the PhysioNet S3 mirror (physionet.org answers 302 there; the project's keep-alive client is
pinned to one host). Same parsing and manifest as ``cardiolens.validation.fetch_ptbdb``; licensed, git-ignored.

    python tools/mi_bench/ptbdb_fetch.py --out data/ptbdb
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cardiolens.validation import fetch_ptbdb as fp


def fetch_bytes(url: str, max_bytes: int | None = None, tries: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


fp.fetch_bytes = fetch_bytes
fp.BASE_URL = "https://physionet-open.s3.amazonaws.com/ptbdb/1.0.0/"

if __name__ == "__main__":
    fp.main()
