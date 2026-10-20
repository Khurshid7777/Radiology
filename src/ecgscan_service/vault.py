"""Encrypted originals (TZ §9, SD-008).

The original bytes are kept only Fernet-encrypted (AES-128-CBC + HMAC, from
``cryptography``) with a key from ``$ECGSCAN_STORAGE_KEY``. Without a key — or
without ``cryptography`` — the vault is absent and originals are simply not kept:
a scan is processed from memory and cannot be re-run after a restart or a
calibration correction. Nothing is ever written in plaintext.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_ENV = "ECGSCAN_STORAGE_KEY"


class Vault:
    def __init__(self, root: Path, key: bytes):
        from cryptography.fernet import Fernet

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key)

    @classmethod
    def from_env(cls, root: Path) -> Vault | None:
        key = os.environ.get(KEY_ENV)
        if not key:
            return None
        try:
            return cls(root, key.encode())
        except ImportError:
            return None

    def _path(self, scan_id: str) -> Path:
        return self.root / f"{scan_id}.bin"

    def put(self, scan_id: str, data: bytes) -> None:
        self._path(scan_id).write_bytes(self._fernet.encrypt(data))

    def get(self, scan_id: str) -> bytes | None:
        p = self._path(scan_id)
        return self._fernet.decrypt(p.read_bytes()) if p.exists() else None

    def has(self, scan_id: str) -> bool:
        return self._path(scan_id).exists()

    def delete(self, scan_id: str) -> None:
        self._path(scan_id).unlink(missing_ok=True)

    # -- artifact images (D98): encrypted beside their scan, read back on request --------------

    def seal_file(self, path: Path) -> Path:
        """Encrypt ``path`` to ``path.enc`` and remove the plaintext."""
        path = Path(path)
        sealed = path.with_name(path.name + ".enc")
        sealed.write_bytes(self._fernet.encrypt(path.read_bytes()))
        path.unlink()
        return sealed

    def open_file(self, sealed: Path) -> bytes:
        return self._fernet.decrypt(Path(sealed).read_bytes())
