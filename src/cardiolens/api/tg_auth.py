"""D350: the site opened from the bot as a Telegram Mini App signs in with Telegram, not with a clinic key.

The page sends Telegram's signed launch data (``initData``) in the ``X-Telegram-Init-Data`` header. The server checks
the signature as Telegram prescribes and lets in only the bot's own users: its administrators
(``ECGSCAN_BOT_ALLOWED``) or those who joined with a live invite code (the bot's access file on the shared volume).

The site never holds the bot's token: only the derived check secret ``ECGSCAN_WEBAPP_SECRET`` =
HMAC-SHA256(key="WebAppData", msg=token), made on the server at deploy — it verifies, it cannot drive the bot.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from urllib.parse import parse_qsl

INIT_HEADER = "x-telegram-init-data"
SECRET_ENV = "ECGSCAN_WEBAPP_SECRET"
MAX_AGE_S = 24 * 3600
FUTURE_SKEW_S = 300
MAX_LEN = 8192
ACTOR_PREFIX = "tg:"


def _secret() -> bytes | None:
    raw = os.environ.get(SECRET_ENV, "").strip()
    try:
        return bytes.fromhex(raw) if re.fullmatch(r"[0-9a-f]{64}", raw) else None
    except ValueError:
        return None


def webapp_secret(bot_token: str) -> str:
    """The check secret for a bot token (what deploy/server_env.sh writes; used by the tests)."""
    return hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).hexdigest()


def telegram_user(init_data: str | None, *, secret: bytes | None = None, now: float | None = None) -> int | None:
    """The Telegram user id of a correctly signed, fresh initData; None otherwise."""
    secret = _secret() if secret is None else secret
    if not secret or not init_data or len(init_data) > MAX_LEN:
        return None
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        return None
    fields = dict(pairs)
    if len(fields) != len(pairs):             # a field given twice: refused, not guessed
        return None
    got = fields.pop("hash", "")
    if not re.fullmatch(r"[0-9a-f]{64}", got):
        return None
    check = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    want = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, got):
        return None
    now = time.time() if now is None else now
    try:
        auth = int(fields.get("auth_date", ""))
        user = json.loads(fields.get("user", ""))
        uid = int(user["id"])
    except (ValueError, TypeError, KeyError):
        return None
    if not (now - MAX_AGE_S <= auth <= now + FUTURE_SKEW_S) or uid <= 0:
        return None
    return uid


def _admins() -> set[int]:
    """The bot's own parser (review: a site-side regex let «77abc» in as 77)."""
    from ecgscan_bot.logic import _ids

    return _ids(os.environ.get("ECGSCAN_BOT_ALLOWED", ""))


def allowed_user(uid: int | None) -> bool:
    """The bot's own user: an administrator, or one who joined with a live invite code (D208)."""
    if uid is None:
        return False
    if uid in _admins():
        return True
    from ecgscan_bot.access import AccessStore

    try:
        return AccessStore().is_member(uid)
    except (AttributeError, TypeError, ValueError):   # a damaged access file refuses, it does not crash (review)
        return False


def telegram_actor(init_data: str | None) -> str | None:
    """«tg:<id>» for an allowed, correctly signed Telegram user; None otherwise."""
    uid = telegram_user(init_data)
    return f"{ACTOR_PREFIX}{uid}" if uid is not None and allowed_user(uid) else None
