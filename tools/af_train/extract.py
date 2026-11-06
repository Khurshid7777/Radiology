"""D205: pull selected Chapman/Ningbo records out of the PhysioNet zip by HTTP range reads.

    python tools/af_train/extract.py --out data/ecg-arrhythmia-extra [--per-class N]

The header-only mirrors under data/ecg-arrhythmia-*/.headers give every record's labels; the
records chosen for training (dump.PER_CLASS, adults) whose signal is not on disk are read from the
zip on the S3 mirror member by member — no full 2 GB download. Licensed data, git-ignored.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump

URL = "https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/ecg-arrhythmia-1.0.0.zip"


class _Ranged(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(head, timeout=60).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", choices=["af", "avb"], default="af", help="af: D205 selection, avb: D206")
    a = ap.parse_args()
    kw = {"codes": dump.AVB_CODES, "per_class": dump.AVB_PER_CLASS} if a.set == "avb" else {}
    want = dump.chapman_list(require_signal=False, **kw)
    missing = {Path(p).name for p, _, _ in want if not Path(p).with_suffix(".mat").exists()}
    print(f"[extract] {len(want)} chosen, {len(missing)} without a signal on disk", flush=True)
    z = zipfile.ZipFile(io.BufferedReader(_Ranged(URL), buffer_size=1 << 20))
    a.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for info in z.infolist():
        name = Path(info.filename)
        if name.suffix in (".hea", ".mat") and name.stem in missing:
            dest = a.out / name.name
            if not dest.exists():
                dest.write_bytes(z.read(info))
            done += 1
            if done % 200 == 0:
                print(f"[extract] {done} files", flush=True)
    print(f"[extract] {done} files", flush=True)


if __name__ == "__main__":
    main()
