"""Analyses kept on disk, encrypted, for 30 days (D199).

The in-memory store loses every analysis on a restart, and with it the browser history, the PDF,
the share links (D197) and the comparison with a prior ECG (D193). Each analysis is written here
Fernet-encrypted with the server's storage key (ECGSCAN_STORAGE_KEY, made on the server, D153),
with the clinic key name that owns it. Without that key nothing is written: memory only, as before.

Only files this server encrypted are unpickled: Fernet authenticates them before anything is read,
so a planted or altered file is refused, not loaded.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

KEY_ENV = "ECGSCAN_STORAGE_KEY"
DIR_ENV = "CARDIOLENS_ANALYSES_DIR"
KEEP_DAYS = 30
_ID = re.compile(r"^[0-9a-f]{12}$")


def _fernet():
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet

        return Fernet(key.encode())
    except Exception:          # a malformed key: say so once per call, keep memory only
        log.warning("%s is not a valid Fernet key: analyses are kept in memory only", KEY_ENV)
        return None


def _dir() -> Path:
    return Path(os.environ.get(DIR_ENV, "/data/analyses"))


def enabled() -> bool:
    return _fernet() is not None


def save(analysis_id: str, result, owner: str | None) -> bool:
    f = _fernet()
    if f is None or not _ID.match(analysis_id):
        return False
    try:
        d = _dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{analysis_id}.tmp"
        tmp.write_bytes(f.encrypt(pickle.dumps({"owner": owner, "result": result})))
        os.chmod(tmp, 0o600)
        tmp.replace(d / f"{analysis_id}.bin")
        purge()
        return True
    except Exception as exc:     # the analysis is still answered from memory
        log.warning("analysis %s not kept on disk: %s", analysis_id, type(exc).__name__)
        return False


def load(analysis_id: str):
    """(result, owner) or None: absent, older than 30 days, altered, or no key."""
    f = _fernet()
    if f is None or not _ID.match(analysis_id):
        return None
    p = _dir() / f"{analysis_id}.bin"
    if not p.exists():
        return None
    try:
        data = f.decrypt(p.read_bytes(), ttl=KEEP_DAYS * 86400)
    except Exception:            # InvalidToken: expired or not ours
        return None
    obj = pickle.loads(data)
    return obj["result"], obj["owner"]


def purge(now: float | None = None) -> int:
    """Remove files older than KEEP_DAYS; returns how many."""
    cutoff = (now or time.time()) - KEEP_DAYS * 86400
    n = 0
    for p in _dir().glob("*.bin"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except OSError:
            pass
    return n
