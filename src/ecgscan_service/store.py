"""SQLite job store and audit log (SD-005, TZ §9 "кто, когда и что подтвердил").

Statuses: ``queued -> processing -> ok | needs_confirmation | rejected | failed``.
The result is the ``ScanResult`` JSON. Nothing identifying the patient is stored
here: the image hash, the client name and the result.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PENDING = ("queued", "processing")
FINAL = ("ok", "needs_confirmation", "rejected", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owner TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    hints TEXT NOT NULL,
    result TEXT,
    error TEXT,
    analysis_id TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ScanRow:
    id: str
    status: str
    created_at: str
    updated_at: str
    owner: str
    sha256: str
    hints: dict
    result: dict | None
    error: str | None
    analysis_id: str | None


class ScanStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.fetchall()

    def create(self, scan_id: str, *, owner: str, sha256: str, hints: dict) -> None:
        t = now()
        self._exec("INSERT INTO scans (id, status, created_at, updated_at, owner, sha256, hints) VALUES (?,?,?,?,?,?,?)",
                   (scan_id, "queued", t, t, owner, sha256, json.dumps(hints)))

    def update(self, scan_id: str, *, status: str, result: dict | None = None, error: str | None = None,
               hints: dict | None = None) -> None:
        sets, args = ["status = ?", "updated_at = ?"], [status, now()]
        if result is not None:
            sets.append("result = ?")
            args.append(json.dumps(result, ensure_ascii=False))
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        if hints is not None:
            sets.append("hints = ?")
            args.append(json.dumps(hints))
        self._exec(f"UPDATE scans SET {', '.join(sets)} WHERE id = ?", (*args, scan_id))

    def set_analysis(self, scan_id: str, analysis_id: str) -> None:
        self._exec("UPDATE scans SET analysis_id = ?, updated_at = ? WHERE id = ?", (analysis_id, now(), scan_id))

    def get(self, scan_id: str) -> ScanRow | None:
        rows = self._exec("SELECT id, status, created_at, updated_at, owner, sha256, hints, result, error, analysis_id "
                          "FROM scans WHERE id = ?", (scan_id,))
        if not rows:
            return None
        r = rows[0]
        return ScanRow(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6]), json.loads(r[7]) if r[7] else None,
                       r[8], r[9])

    def pending(self) -> list[str]:
        return [r[0] for r in self._exec("SELECT id FROM scans WHERE status IN (?, ?) ORDER BY created_at", PENDING)]

    def older_than(self, iso: str) -> list[str]:
        return [r[0] for r in self._exec("SELECT id FROM scans WHERE created_at < ?", (iso,))]

    def scrub_comments(self, ids: list[str]) -> None:
        """Free-text doctor comments of expired scans are removed with the images (D98)."""
        for scan_id in ids:
            self._exec("UPDATE audit SET details = json_remove(details, '$.comment') "
                       "WHERE scan_id = ? AND details IS NOT NULL AND json_valid(details)", (scan_id,))

    def audit(self, scan_id: str, actor: str, action: str, details: dict | None = None) -> None:
        self._exec("INSERT INTO audit (scan_id, at, actor, action, details) VALUES (?,?,?,?,?)",
                   (scan_id, now(), actor, action, json.dumps(details, ensure_ascii=False) if details else None))

    def audit_log(self, scan_id: str) -> list[dict]:
        rows = self._exec("SELECT at, actor, action, details FROM audit WHERE scan_id = ? ORDER BY seq", (scan_id,))
        return [{"at": a, "actor": b, "action": c, "details": json.loads(d) if d else None} for a, b, c, d in rows]
