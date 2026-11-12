"""Who may use the bot (D208): administrators from the environment, others by an invite code.

The store is one JSON file on the app's volume. It keeps the Telegram id, the code a user joined with,
the time and the chosen reply language — no names, no messages, no ECG data. SD-107: also the groups
an administrator switched on for reading (chat id, title, who and when).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path

STORE_ENV = "ECGSCAN_BOT_STORE"
DEFAULT_STORE = "/data/bot/access.json"
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # no 0/O, 1/I: codes are typed by hand too
CODE_LEN = 8


class AccessStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get(STORE_ENV, DEFAULT_STORE))
        self._lock = threading.Lock()
        self._data = {"codes": {}, "users": {}, "groups": {}}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        self._data.setdefault("codes", {})
        self._data.setdefault("users", {})
        self._data.setdefault("groups", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    # --- codes (administrators) ---
    def new_code(self, label: str, by: int) -> str:
        with self._lock:
            while True:
                code = "".join(secrets.choice(_ALPHABET) for _ in range(CODE_LEN))
                if code not in self._data["codes"]:
                    break
            self._data["codes"][code] = {"label": (label or "")[:60], "by": by, "at": int(time.time()),
                                         "revoked": False}
            self._save()
            return code

    def revoke(self, code: str) -> bool:
        code = (code or "").strip().upper()
        with self._lock:
            c = self._data["codes"].get(code)
            if c is None:
                return False
            c["revoked"] = True
            self._save()
            return True

    def codes(self) -> list[dict]:
        users = self._data["users"].values()
        return [{"code": k, **v, "users": sum(1 for u in users if u.get("code") == k)}
                for k, v in sorted(self._data["codes"].items(), key=lambda kv: kv[1]["at"])]

    # --- users ---
    def join(self, user_id: int, code: str) -> bool:
        code = (code or "").strip().upper()
        with self._lock:
            c = self._data["codes"].get(code)
            if c is None or c.get("revoked"):
                return False
            u = self._data["users"].setdefault(str(user_id), {})
            u.update(code=code, at=int(time.time()))
            self._save()
            return True

    def is_member(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        u = self._data["users"].get(str(user_id))
        if not u or not u.get("code"):
            return False
        c = self._data["codes"].get(u["code"])
        return bool(c) and not c.get("revoked")

    def lang(self, user_id: int | None) -> str | None:
        u = self._data["users"].get(str(user_id)) if user_id is not None else None
        return u.get("lang") if u else None

    def set_lang(self, user_id: int, lang: str) -> None:
        if lang not in ("ru", "uz"):
            return
        with self._lock:
            self._data["users"].setdefault(str(user_id), {})["lang"] = lang
            self._save()

    # --- groups (SD-107) ---
    def set_group(self, chat_id: int, title: str | None, on: bool, by: int | None = None) -> None:
        with self._lock:
            g = self._data["groups"].setdefault(str(chat_id), {})
            g.update(on=bool(on), at=int(time.time()))
            if title:
                g["title"] = title[:80]
            if by is not None:
                g["by"] = by
            self._save()

    def group_on(self, chat_id: int | None) -> bool:
        g = self._data["groups"].get(str(chat_id)) if chat_id is not None else None
        return bool(g and g.get("on"))

    def groups(self) -> list[dict]:
        return [{"chat_id": int(k), **v} for k, v in sorted(self._data["groups"].items())]
