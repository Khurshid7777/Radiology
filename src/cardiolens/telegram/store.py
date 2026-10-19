"""Append-only dataset of collected photos, predictions and reviewer verdicts.

One JSONL line per upsert (the last line for an id wins), so the file is a full
audit trail: the photo, what the pipeline said, what the reviewer answered.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field


class Record(BaseModel):
    id: str                       # f"{chat_id}_{msg_id}"
    chat_id: int
    msg_id: int
    chat_title: str = ""
    date: str = ""
    sender_hash: str = ""         # sha256 prefix of the sender id — never the username
    caption: str = ""
    photo_path: str = ""
    signal_path: str | None = None
    review_png: str | None = None
    status: str = "new"           # not_ecg | digitise_failed | analysed | merged (another album photo's record)
    grid_pitch_px: float | None = None
    error: str | None = None
    predicted: list[str] = Field(default_factory=list)
    critical: list[str] = Field(default_factory=list)
    confidences: dict[str, float] = Field(default_factory=dict)
    measurements: dict = Field(default_factory=dict)
    analysis_mode: str | None = None
    provider: str | None = None
    review_msg_ids: list[int] = Field(default_factory=list)
    verdict: str | None = None    # confirmed | rejected | partial | comment
    gold: list[str] | None = None
    comment: str | None = None
    verdict_time: str | None = None

    @property
    def verified(self) -> bool:
        return self.gold is not None


class DatasetStore:
    """``store_dir/dataset.jsonl`` plus the photo / signal folders next to it."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "dataset.jsonl"
        self.photos = self.root / "photos"
        self.signals = self.root / "signals"
        self.reviews = self.root / "reviews"
        self._records: dict[str, Record] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = Record(**json.loads(line))
                self._records[rec.id] = rec

    @staticmethod
    def record_id(chat_id: int, msg_id: int) -> str:
        return f"{chat_id}_{msg_id}"

    def has(self, chat_id: int, msg_id: int) -> bool:
        return self.record_id(chat_id, msg_id) in self._records

    def get(self, rec_id: str) -> Record | None:
        return self._records.get(rec_id)

    def upsert(self, rec: Record) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
        self._records[rec.id] = rec

    def all(self) -> list[Record]:
        return list(self._records.values())

    def by_review_msg(self, msg_id: int) -> Record | None:
        for rec in self._records.values():
            if msg_id in rec.review_msg_ids:
                return rec
        return None

    def stats(self) -> dict:
        recs = self.all()
        status = Counter(r.status for r in recs)
        verdict = Counter(r.verdict for r in recs if r.verdict)
        return {
            "total": len(recs),
            "not_ecg": status.get("not_ecg", 0),
            "digitise_failed": status.get("digitise_failed", 0),
            "analysed": status.get("analysed", 0),
            "awaiting_review": sum(1 for r in recs if r.status not in ("not_ecg", "merged") and r.verdict is None),
            "confirmed": verdict.get("confirmed", 0),
            "rejected": verdict.get("rejected", 0),
            "partial": verdict.get("partial", 0),
            "comment": verdict.get("comment", 0),
            "verified": sum(1 for r in recs if r.verified),
        }
