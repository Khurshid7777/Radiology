"""Mini App capture (SD-137): one-time tickets and the in-memory inbox between the page and the bot.

A photo taken on the Mini App page goes to the server as the file the phone made, so Telegram
never compresses it. The page has no clinic key: it carries a ticket the bot issued. A ticket is
``<id>.<expiry>.<signature>``: a random id, a Unix time and an HMAC-SHA256 made with the bot's API
key, which the server checks against its own keys. It holds no Telegram id; whom to answer only the
bot knows. The uploaded file waits in memory (never on disk) until the bot takes it or the ticket
expires. One server process is assumed (as the image runs it): the inbox is not shared.

Tickets are checked by signature, not kept: a restart of the server loses the waiting files and the
list of used tickets, never the validity of a live ticket (the page can simply upload again).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from dataclasses import dataclass

TICKET_HEADER = "X-Miniapp-Ticket"
_TICKET = re.compile(r"^([0-9a-f]{24})\.([0-9]{1,12})\.([0-9a-f]{64})$")
_DOMAIN = "miniapp1"            # what the signature is for: a ticket is not valid as anything else


def _sign(key: str, tid: str, expiry: int) -> str:
    return hmac.new(key.encode(), f"{_DOMAIN}|{tid}|{expiry}".encode(), hashlib.sha256).hexdigest()


def issue_ticket(key: str, minutes: float, now: float | None = None) -> tuple[str, str, int]:
    """(ticket, its id, its expiry) signed with ``key``."""
    tid = os.urandom(12).hex()
    expiry = int((time.time() if now is None else now) + minutes * 60)
    return f"{tid}.{expiry}.{_sign(key, tid, expiry)}", tid, expiry


@dataclass(frozen=True)
class Ticket:
    id: str
    owner: str          # the name of the API key that signed it
    expiry: int


def verify_ticket(ticket: str | None, keys: dict[str, str], *, max_minutes: float,
                  now: float | None = None) -> Ticket | None:
    """The live ticket signed with one of ``keys`` (key -> name), or None. A ticket valid for longer
    than ``max_minutes`` from now is refused too: a signer cannot mint a ticket for ever."""
    m = _TICKET.match(ticket or "")
    if m is None:
        return None
    tid, expiry, sig = m.group(1), int(m.group(2)), m.group(3)
    now = time.time() if now is None else now
    owner = None
    for key, name in keys.items():       # every key compared: no early exit on a match
        if hmac.compare_digest(_sign(key, tid, expiry), sig):
            owner = name
    if owner is None or expiry <= now or expiry > now + max_minutes * 60:
        return None
    return Ticket(tid, owner, expiry)


def image_kind(data: bytes) -> str | None:
    """What the first bytes say the file is (never the name or the type the client declared)."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return None


class InboxFull(Exception):
    pass


@dataclass
class _Item:
    data: bytes
    owner: str
    expiry: int
    ready: bool = False


class Inbox:
    """Uploaded files waiting for the bot, in memory only."""

    def __init__(self, max_waiting: int):
        self.max_waiting = max_waiting
        self._lock = threading.Lock()
        self._items: dict[str, _Item] = {}
        self._used: dict[str, int] = {}          # ticket id -> expiry: taken by the bot, never accepted again
        self._tries: dict[str, list[int]] = {}   # ticket id -> [expiry, uploads so far]

    def _purge(self, now: float) -> None:
        for d, expiry in ((self._items, lambda v: v.expiry), (self._used, lambda v: v), (self._tries, lambda v: v[0])):
            for k in [k for k, v in d.items() if expiry(v) <= now]:
                del d[k]

    def attempt(self, ticket: Ticket, limit: int, now: float | None = None) -> bool:
        """Count one upload of the ticket; False when it has had ``limit`` uploads or its file is taken."""
        with self._lock:
            self._purge(time.time() if now is None else now)
            tries = self._tries.setdefault(ticket.id, [ticket.expiry, 0])
            if ticket.id in self._used or tries[1] >= limit:
                return False
            tries[1] += 1
            return True

    def used(self, tid: str, now: float | None = None) -> bool:
        with self._lock:
            self._purge(time.time() if now is None else now)
            return tid in self._used

    def put(self, ticket: Ticket, data: bytes, now: float | None = None) -> None:
        """Keep the file for the ticket; a new file for the same ticket replaces the earlier one."""
        with self._lock:
            self._purge(time.time() if now is None else now)
            if ticket.id in self._used:
                raise KeyError(ticket.id)
            if ticket.id not in self._items and len(self._items) >= self.max_waiting:
                raise InboxFull
            self._items[ticket.id] = _Item(data, ticket.owner, ticket.expiry)

    def send(self, ticket: Ticket, now: float | None = None) -> bool:
        """Mark the ticket's file ready for the bot; False when nothing is waiting."""
        with self._lock:
            self._purge(time.time() if now is None else now)
            item = self._items.get(ticket.id)
            if item is None:
                return False
            item.ready = True
            return True

    def take(self, tid: str, owner: str, now: float | None = None) -> bytes | None | bool:
        """The ready file of the ticket, once: bytes; None: nothing ready yet; False: already taken.
        Only the key that signed the ticket gets it."""
        with self._lock:
            self._purge(time.time() if now is None else now)
            if tid in self._used:
                return False
            item = self._items.get(tid)
            if item is None or not item.ready or item.owner != owner:
                return None
            del self._items[tid]
            self._used[tid] = item.expiry
            return item.data

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
