"""SD-152: the desk agent's key and the queue of records it sent, waiting for the bot.

A doctor's laptop runs the agent next to the recorder's own PC program; the agent sends each record the program saves.
The agent holds an **agent key**: ``<agent id>.<expiry>.<signature>``, HMAC-SHA256 with the bot's API key, as the Mini App
tickets (SD-137) but for another purpose (a ticket is never an agent key, nor the other way round). The key can only put
files into the queue; the bot alone knows which chat an agent id belongs to. Files wait in memory, never on disk.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

AGENT_HEADER = "X-Agent-Key"
_KEY = re.compile(r"^([0-9a-f]{16})\.([0-9]{1,12})\.([0-9a-f]{64})$")
_DOMAIN = "agent1"
KINDS = {                                   # what a file is, by its first bytes (never by its name)
    "jpeg": lambda b: b[:3] == b"\xff\xd8\xff",
    "png": lambda b: b[:8] == b"\x89PNG\r\n\x1a\n",
    "tiff": lambda b: b[:4] in (b"II*\x00", b"MM\x00*"),
    "pdf": lambda b: b[:5] == b"%PDF-",
    "xml": lambda b: b.lstrip(b"\xef\xbb\xbf \t\r\n")[:1] == b"<",
    "dicom": lambda b: b[128:132] == b"DICM",
}


def kind_of(head: bytes, allowed: list[str]) -> str | None:
    """The first of ``allowed`` kinds the bytes are, or None."""
    return next((k for k in allowed if k in KINDS and KINDS[k](head)), None)


def _sign(key: str, aid: str, expiry: int) -> str:
    return hmac.new(key.encode(), f"{_DOMAIN}|{aid}|{expiry}".encode(), hashlib.sha256).hexdigest()


def new_agent_id() -> str:
    return os.urandom(8).hex()


def issue_agent_key(key: str, agent_id: str, days: float, now: float | None = None) -> tuple[str, int]:
    """(agent key, its expiry) for ``agent_id``, signed with ``key`` (the bot's API key)."""
    expiry = int((time.time() if now is None else now) + days * 86400)
    return f"{agent_id}.{expiry}.{_sign(key, agent_id, expiry)}", expiry


@dataclass(frozen=True)
class AgentKey:
    id: str
    owner: str          # the name of the API key that signed it (the bot's)
    expiry: int


def verify_agent_key(raw: str | None, keys: dict[str, str], *, max_days: float, now: float | None = None,
                     refused_prefix: str | None = "agent") -> AgentKey | None:
    """The live agent key signed with one of ``keys`` (key -> name), or None; a read-only key never signs one."""
    m = _KEY.match(raw or "")
    if m is None:
        return None
    aid, expiry, sig = m.group(1), int(m.group(2)), m.group(3)
    now = time.time() if now is None else now
    owner = None
    for key, name in keys.items():       # every key compared: no early exit on a match
        if refused_prefix and name.startswith(refused_prefix):
            continue
        if hmac.compare_digest(_sign(key, aid, expiry), sig):
            owner = name
    if owner is None or expiry <= now or expiry > now + max_days * 86400:
        return None
    return AgentKey(aid, owner, expiry)


class QueueFull(Exception):
    pass


@dataclass
class _Record:
    agent: str
    owner: str
    kind: str
    data: bytes
    until: float


class AgentQueue:
    """Records from the agents, oldest first, in memory only; each waits at most ``keep_s``."""

    def __init__(self, *, max_items: int, max_bytes: int, per_agent: int, keep_s: float):
        self.max_items, self.max_bytes, self.per_agent, self.keep_s = max_items, max_bytes, per_agent, keep_s
        self._lock = threading.Lock()
        self._q: deque[_Record] = deque()

    def _purge(self, now: float) -> None:
        while self._q and self._q[0].until <= now:
            self._q.popleft()
        if any(r.until <= now for r in self._q):
            self._q = deque(r for r in self._q if r.until > now)

    def room(self, agent: AgentKey, size: int, now: float | None = None) -> bool:
        """Is there room for one more record of ``size`` bytes from this agent (checked before its body is read)?"""
        with self._lock:
            self._purge(time.time() if now is None else now)
            return (len(self._q) < self.max_items and sum(1 for r in self._q if r.agent == agent.id) < self.per_agent
                    and sum(len(r.data) for r in self._q) + size <= self.max_bytes)

    def put(self, agent: AgentKey, kind: str, data: bytes, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._purge(now)
            if (len(self._q) >= self.max_items or sum(1 for r in self._q if r.agent == agent.id) >= self.per_agent
                    or sum(len(r.data) for r in self._q) + len(data) > self.max_bytes):
                raise QueueFull
            self._q.append(_Record(agent.id, agent.owner, kind, data, now + self.keep_s))

    def take(self, owner: str, now: float | None = None) -> tuple[str, str, bytes] | None:
        """The oldest record for the key ``owner``: (agent id, kind, bytes), once; None when there is none."""
        with self._lock:
            self._purge(time.time() if now is None else now)
            for r in self._q:
                if r.owner == owner:
                    self._q.remove(r)
                    return r.agent, r.kind, r.data
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)
