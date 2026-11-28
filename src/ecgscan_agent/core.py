"""SD-152: the desk agent. Standard library only (one .exe for Windows later).

It watches the folders where the recorder's own PC program (Edan Smart ECG Viewer, Contec ECG-Sync, ...) saves records,
waits until a new file stops growing, and sends it once to the CardioVision server with the agent key the bot gave the
doctor (/agent). Sent files are remembered by their SHA-256, so a restart sends nothing twice; when the server cannot be
reached the file is sent on a later round. On the first run the files already in the folders are taken as sent, so an old
archive does not flood the doctor's chat. Logs never hold a file name or a byte of a record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("ecgscan_agent")
UPLOAD_PATH = "/miniapp/agent/upload"
KEY_HEADER = "X-Agent-Key"
RETRY = {0, 408, 429, 500, 502, 503, 504}      # tried again on a later round; anything else is final for that file
DEFAULT_EXTS = [".pdf", ".xml", ".dcm", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def home() -> Path:
    """Where the agent keeps its settings and memory."""
    base = os.environ.get("APPDATA") if sys.platform == "win32" else os.environ.get("XDG_CONFIG_HOME")
    return Path(base or Path.home() / ".config") / "cardiovision-agent"


@dataclass
class Settings:
    server: str
    key: str
    folders: list[str]
    exts: list[str] = field(default_factory=lambda: list(DEFAULT_EXTS))
    poll_s: float = 3.0
    stable_s: float = 2.0             # a file is sent once its size has not changed for this long
    max_mb: float = 25.0
    timeout_s: float = 60.0

    def check(self) -> None:
        if not self.server.startswith("https://") and not self.server.startswith("http://127.0.0.1"):
            raise ValueError("server: an https:// address")
        if self.key.count(".") != 2:
            raise ValueError("key: the line the bot gave on /agent")
        if not self.folders:
            raise ValueError("folders: at least one")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        if sys.platform != "win32":
            os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: Path) -> Settings:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class Memory:
    """What was sent (or refused for good), by SHA-256; kept in a small JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict] = {}
        if path.exists():
            try:
                self.done = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.done = {}

    def has(self, digest: str) -> bool:
        return digest in self.done

    def mark(self, digest: str, outcome: str) -> None:
        self.done[digest] = {"outcome": outcome, "at": int(time.time())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.done), encoding="utf-8")
        tmp.replace(self.path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(folders: list[str], exts: list[str]) -> list[Path]:
    """Files of the allowed kinds in the folders (and below), oldest first."""
    want = {e.lower() for e in exts}
    out = []
    for folder in folders:
        root = Path(folder)
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in want:
                    out.append((p.stat().st_mtime, p))
            except OSError:
                continue
    return [p for _, p in sorted(out)]


def send(settings: Settings, data: bytes) -> int:
    """POST one record; the HTTP status (0: the server could not be reached)."""
    req = urllib.request.Request(settings.server.rstrip("/") + UPLOAD_PATH, data=data, method="POST",
                                 headers={KEY_HEADER: settings.key, "Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout_s) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0


class Agent:
    def __init__(self, settings: Settings, memory: Memory, *, sender=send, clock=time.monotonic):
        self.s, self.mem, self.sender, self.clock = settings, memory, sender, clock
        self.sizes: dict[Path, tuple[int, float]] = {}      # path -> (size, since when it is that size)

    def baseline(self) -> int:
        """First run: what is in the folders already counts as sent (an old archive must not flood the chat)."""
        n = 0
        for p in candidates(self.s.folders, self.s.exts):
            try:
                d = sha256(p)
            except OSError:
                continue
            if not self.mem.has(d):
                self.mem.mark(d, "baseline")
                n += 1
        return n

    def round(self) -> dict:
        """One look at the folders: send what is new and settled. Counts of what happened."""
        now = self.clock()
        out = {"sent": 0, "retry": 0, "refused": 0, "waiting": 0}
        seen = set()
        for p in candidates(self.s.folders, self.s.exts):
            seen.add(p)
            try:
                size = p.stat().st_size
            except OSError:
                continue
            last = self.sizes.get(p)
            if last is None or last[0] != size:
                self.sizes[p] = (size, now)
                out["waiting"] += 1
                continue
            if now - last[1] < self.s.stable_s or size == 0:
                out["waiting"] += 1
                continue
            try:
                digest = sha256(p)
                if self.mem.has(digest):
                    continue
                if size > self.s.max_mb * 1024 * 1024:
                    self.mem.mark(digest, "too_large")
                    out["refused"] += 1
                    log.warning("a record %s… over %.0f MB, not sent", digest[:8], self.s.max_mb)
                    continue
                data = p.read_bytes()
            except OSError:
                continue
            status = self.sender(self.s, data)
            if status == 200:
                self.mem.mark(digest, "sent")
                out["sent"] += 1
                log.info("record %s… sent (%d bytes)", digest[:8], len(data))
            elif status in RETRY:
                out["retry"] += 1
                log.warning("record %s… not sent now (%s), again later", digest[:8], status or "no connection")
            else:
                self.mem.mark(digest, f"refused_{status}")
                out["refused"] += 1
                log.warning("record %s… refused by the server (%d)", digest[:8], status)
        for p in [p for p in self.sizes if p not in seen]:
            del self.sizes[p]
        return out

    def run(self, stop=lambda: False) -> None:
        while not stop():
            try:
                self.round()
            except Exception:
                log.exception("a round failed")
            time.sleep(self.s.poll_s)
