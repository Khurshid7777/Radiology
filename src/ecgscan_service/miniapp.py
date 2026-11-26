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
                  now: float | None = None, refused_prefix: str | None = "agent") -> Ticket | None:
    """The live ticket signed with one of ``keys`` (key -> name), or None. A ticket valid for longer
    than ``max_minutes`` from now is refused too: a signer cannot mint a ticket for ever."""
    m = _TICKET.match(ticket or "")
    if m is None:
        return None
    tid, expiry, sig = m.group(1), int(m.group(2)), m.group(3)
    now = time.time() if now is None else now
    owner = None
    for key, name in keys.items():       # every key compared: no early exit on a match
        if refused_prefix and name.startswith(refused_prefix):
            continue                      # a read-only key (the core's agents) never signs a ticket
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


TURNS = (0, 90, 180, 270)


def size_refusal(data: bytes) -> str | None:
    """Why the picture is not decoded (None: it may be): a 1-MB PNG can declare 30000 x 30000 px and
    take gigabytes to decode (D136), so the header is read first, with the limits of the analysis. A header
    that cannot be read is refused too: some TIFF headers PIL does not read expand to gigabytes in OpenCV."""
    from ecgscan.config import load_yaml
    from ecgscan.io import ImageRefused, check_image, image_size

    if image_size(data) is None:          # a header the reader does not know: "the decoder decides" would decode it
        return "UNREADABLE_HEADER"
    q = load_yaml("pipeline")["quality"]
    try:
        check_image(data, max_megapixels=float(q["max_megapixels"]), min_side_px=int(q["min_side_px"]))
    except ImageRefused as exc:
        return exc.code
    return None


def turn_photo(data: bytes, rotate_deg: int, quality: int, max_bytes: int) -> bytes:
    """The photo turned ``rotate_deg`` clockwise (SD-138), as a JPEG: decoded as the doctor saw it
    (EXIF applied), turned, encoded at ``quality`` (lower if it would not fit ``max_bytes``).
    Raises ValueError when it cannot be done."""
    import cv2
    import numpy as np

    if rotate_deg not in TURNS:
        raise ValueError(f"not a quarter turn: {rotate_deg}")
    if rotate_deg == 0:
        return data
    why = size_refusal(data)
    if why:
        raise ValueError("the picture is not decoded: " + why)
    try:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("not a picture") from exc
    if bgr is None:
        raise ValueError("not a picture")
    bgr = np.ascontiguousarray(np.rot90(bgr, k=-rotate_deg // 90))
    for q in range(int(quality), 49, -10):
        ok, out = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and out.size <= max_bytes:
            return out.tobytes()
    raise ValueError("the turned photo does not fit the upload limit")


class InboxFull(Exception):
    """The whole store is full (all tickets together): try again later."""


class TooMuch(Exception):
    """This ticket already holds as many photos or bytes as one record may have."""


@dataclass
class _Item:
    owner: str
    expiry: int
    parts: list[bytes]
    turns: list[int]    # the doctor's quarter turn of each part, clockwise (SD-138); applied when the bot takes them
    ready: bool = False


class Inbox:
    """Uploaded photos waiting for the bot, in memory only. SD-147: a ticket holds the photos of one record, in order
    (a long strip shot in parts); the bot takes them all at once."""

    def __init__(self, max_waiting: int, *, max_files: int = 10, max_ticket_bytes: int = 60 << 20,
                 max_store_bytes: int = 300 << 20, flying_max_s: float = 900.0):
        self.max_waiting, self.max_files = max_waiting, max_files
        self.max_ticket_bytes, self.max_store_bytes = max_ticket_bytes, max_store_bytes
        self.flying_max_s = flying_max_s
        self._lock = threading.Lock()
        self._items: dict[str, _Item] = {}
        self._used: dict[str, int] = {}          # ticket id -> expiry: taken by the bot, never accepted again
        self._tries: dict[str, list[int]] = {}   # ticket id -> [expiry, uploads so far]
        self._flying: dict[str, list[int]] = {}  # ticket id -> bytes reserved by uploads still being read
        self._flying_until: dict[str, float] = {}  # ticket id -> when its reservations go even if never given back

    def _purge(self, now: float) -> None:
        # An upload gives its room back itself (add, or release in its finally) whatever its ticket's end: the frame check
        # may wait in its pool (review of D354, SD-155). The time bound only covers a bug: flying_max_s from the reservation.
        for d, expiry in ((self._items, lambda v: v.expiry), (self._used, lambda v: v), (self._tries, lambda v: v[0]),
                          (self._flying_until, lambda v: v)):
            for k in [k for k, v in d.items() if expiry(v) <= now]:
                del d[k]
        for k in [k for k in self._flying if k not in self._flying_until]:
            del self._flying[k]

    def _now(self, now: float | None) -> None:
        self._purge(time.time() if now is None else now)

    def attempt(self, ticket: Ticket, limit: int, now: float | None = None) -> bool:
        """Count one upload of the ticket; False when it has had ``limit`` uploads or its photos are taken."""
        with self._lock:
            self._now(now)
            tries = self._tries.setdefault(ticket.id, [ticket.expiry, 0])
            if ticket.id in self._used or tries[1] >= limit:
                return False
            tries[1] += 1
            return True

    def used(self, tid: str, now: float | None = None) -> bool:
        with self._lock:
            self._now(now)
            return tid in self._used

    def reserve(self, ticket: Ticket, declared: int | None, file_cap: int, max_inflight: int,
                now: float | None = None, per_ticket: int = 2) -> int:
        """Before a byte of the body is read (the core's review, SD-147): room for one more photo of this record. Returns
        how many bytes this upload may read (the declared size, or what the record and the store have left, at most
        ``file_cap``) and keeps them reserved until ``add`` or ``release``. KeyError: the ticket is spent; TooMuch: the
        record is full; InboxFull: the store is, or ``max_inflight`` uploads are being read already."""
        with self._lock:
            self._now(now)
            if ticket.id in self._used:
                raise KeyError(ticket.id)
            if sum(len(v) for v in self._flying.values()) >= max_inflight:
                raise InboxFull
            item = self._items.get(ticket.id)
            flying = self._flying.get(ticket.id, [])
            if len(flying) >= per_ticket:     # one button cannot hold the service's uploads (the core's review)
                raise InboxFull
            if (len(item.parts) if item else 0) + len(flying) >= self.max_files:
                raise TooMuch
            held = (sum(len(b) for b in item.parts) if item else 0) + sum(flying)
            left = self.max_ticket_bytes - held
            want = min(file_cap, left) if declared is None else declared
            if left <= 0 or want > left:
                raise TooMuch
            store = sum(len(b) for it in self._items.values() for b in it.parts) + sum(sum(v) for v in self._flying.values())
            if store + want > self.max_store_bytes:
                raise InboxFull
            if item is None and not flying and len(self._items) + len(self._flying) >= self.max_waiting:
                raise InboxFull
            self._flying.setdefault(ticket.id, []).append(want)
            self._flying_until[ticket.id] = (time.time() if now is None else now) + self.flying_max_s
            return want

    def release(self, ticket: Ticket, reserved: int) -> None:
        with self._lock:
            self._release(ticket.id, reserved)

    def _release(self, tid: str, reserved: int) -> None:
        flying = self._flying.get(tid)
        if flying and reserved in flying:
            flying.remove(reserved)
            if not flying:
                del self._flying[tid]
                self._flying_until.pop(tid, None)

    def add(self, ticket: Ticket, data: bytes, now: float | None = None, reserved: int | None = None) -> int:
        """Keep one more photo of the ticket's record; its number (0, 1, ...). KeyError: the ticket is spent;
        TooMuch: the record is full; InboxFull: the store is. ``reserved``: what ``reserve`` kept for it, given back."""
        with self._lock:
            if reserved is not None:
                self._release(ticket.id, reserved)
            self._now(now)
            if ticket.id in self._used:
                raise KeyError(ticket.id)
            item = self._items.get(ticket.id)
            if item is None and len(self._items) >= self.max_waiting:
                raise InboxFull
            held = sum(len(b) for b in item.parts) if item else 0
            if item is not None and (len(item.parts) >= self.max_files or held + len(data) > self.max_ticket_bytes):
                raise TooMuch
            if len(data) > self.max_ticket_bytes:
                raise TooMuch
            if sum(len(b) for it in self._items.values() for b in it.parts) + len(data) > self.max_store_bytes:
                raise InboxFull
            if item is None:
                item = self._items[ticket.id] = _Item(ticket.owner, ticket.expiry, [], [])
            item.parts.append(data)
            item.turns.append(0)
            item.ready = False
            return len(item.parts) - 1

    def put(self, ticket: Ticket, data: bytes, now: float | None = None) -> None:
        """One photo for the ticket, replacing whatever it held (the single-photo flow of SD-137)."""
        with self._lock:
            item = self._items.get(ticket.id)
            if item is not None:
                item.parts, item.turns = [], []
        self.add(ticket, data, now)

    def turn(self, ticket: Ticket, part: int, rotate: int, now: float | None = None) -> bool:
        with self._lock:
            self._now(now)
            item = self._items.get(ticket.id)
            if item is None or not 0 <= part < len(item.parts):
                return False
            item.turns[part] = rotate
            return True

    def drop(self, ticket: Ticket, part: int, now: float | None = None) -> bool:
        with self._lock:
            self._now(now)
            item = self._items.get(ticket.id)
            if item is None or not 0 <= part < len(item.parts):
                return False
            del item.parts[part], item.turns[part]
            if not item.parts:
                del self._items[ticket.id]
            return True

    def send(self, ticket: Ticket, now: float | None = None, rotate: int = 0) -> bool:
        """Mark the ticket's photos ready for the bot; ``rotate`` (the SD-138 form) turns the last one. False when
        nothing is waiting."""
        with self._lock:
            self._now(now)
            item = self._items.get(ticket.id)
            if item is None or not item.parts:
                return False
            if rotate:
                item.turns[-1] = rotate
            item.ready = True
            return True

    def take(self, tid: str, owner: str, now: float | None = None, *, with_turn: bool = False):
        """The ready photos of the ticket, once and in order: a list of bytes (with ``with_turn``: of (bytes, turn));
        None: nothing ready yet; False: already taken. Only the key that signed the ticket gets them."""
        with self._lock:
            self._now(now)
            if tid in self._used:
                return False
            item = self._items.get(tid)
            if item is None or not item.ready or item.owner != owner:
                return None
            del self._items[tid]
            self._used[tid] = item.expiry
            return list(zip(item.parts, item.turns, strict=True)) if with_turn else list(item.parts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
