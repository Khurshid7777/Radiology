"""The bot's small state (D279): the bound supergroup, topic ids, Claude sessions per topic, a pending deploy."""
from __future__ import annotations

import json
import os
from pathlib import Path


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = {"chat_id": None, "topics": {}, "sessions": {}, "pending": {}, "paused": False,
                           "proposals": {}, "last_reports": {}}
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text()))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()
