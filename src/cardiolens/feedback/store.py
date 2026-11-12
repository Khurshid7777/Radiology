"""Physician feedback capture (TZ §13).

Captures structured physician corrections — for each finding/chain: confirmed,
corrected, or rejected — into an append-only JSONL log. This is the fuel for
future fine-tuning and, crucially, for fitting the confidence calibrator
(TZ §13): each confirmed/rejected finding is a (confidence, label) sample.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["confirmed", "corrected", "rejected"]
COMMENT_MAX = 300        # D260: the bot sends <= 300 characters; the server enforces it
SOURCE_MAX = 32
_LONG_DIGITS = re.compile(r"\d{5,}")    # phone, card, policy numbers: not ECG facts


def clean_comment(text: str | None) -> str | None:
    """D260: a doctor's free text, capped and with long digit runs masked."""
    if text is None:
        return None
    text = _LONG_DIGITS.sub("…", text.strip())[:COMMENT_MAX]
    return text or None


class FindingFeedback(BaseModel):
    code: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)   # model confidence at review time
    corrected_code: str | None = None            # set when verdict == "corrected"
    comment: str | None = None


class Feedback(BaseModel):
    """One physician review of an analysis (TZ §13)."""

    analysis_id: str
    reviewer: str | None = None
    mode: str | None = None                      # mode the system was in
    findings: list[FindingFeedback] = Field(default_factory=list)
    overall_agreement: bool | None = None
    timestamp: float = Field(default_factory=lambda: time.time())
    # D260: the doctor's "what it really is" and where the verdict came from (web / bot / bot-group).
    comment: str | None = None
    source: str | None = None

    @field_validator("comment")
    @classmethod
    def _clean_comment(cls, v: str | None) -> str | None:
        return clean_comment(v)

    @field_validator("source")
    @classmethod
    def _clean_source(cls, v: str | None) -> str | None:
        return v.strip()[:SOURCE_MAX] or None if v is not None else None

    def calibration_samples(self) -> list[tuple[float, int]]:
        """(confidence, label) pairs for calibrator fitting (TZ §13, §16).

        confirmed -> 1, rejected -> 0. 'corrected' is ambiguous and skipped.
        """
        out: list[tuple[float, int]] = []
        for f in self.findings:
            if f.verdict == "confirmed":
                out.append((f.confidence, 1))
            elif f.verdict == "rejected":
                out.append((f.confidence, 0))
        return out


@dataclass
class FeedbackStore:
    """Append-only JSONL feedback log."""

    path: Path
    _cache: list[Feedback] = field(default_factory=list)

    @classmethod
    def at(cls, path: str | Path) -> FeedbackStore:
        return cls(path=Path(path))

    def append(self, fb: Feedback) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fb.model_dump(), ensure_ascii=False) + "\n")

    def all(self) -> list[Feedback]:
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Feedback(**json.loads(line)))
        return out

    def calibration_dataset(self) -> list[tuple[float, int]]:
        """Aggregate all feedback into calibrator training samples (TZ §13)."""
        samples: list[tuple[float, int]] = []
        for fb in self.all():
            samples.extend(fb.calibration_samples())
        return samples


def summarize_feedback(items: list[Feedback], since_ts: float = 0) -> dict:
    """D260: verdict counts for the period, by source, and the disagreements with their comment.
    No reviewer, image, patient field or conclusion text."""
    items = [f for f in items if f.timestamp >= since_ts]
    out: dict = {"n": len(items), "agree": 0, "disagree": 0, "unrated": 0, "by_source": {}, "disagreed": []}
    for f in items:
        key = "agree" if f.overall_agreement is True else "disagree" if f.overall_agreement is False else "unrated"
        out[key] += 1
        s = out["by_source"].setdefault(f.source or "web", {"n": 0, "agree": 0, "disagree": 0, "unrated": 0})
        s["n"] += 1
        s[key] += 1
        if f.overall_agreement is False:
            out["disagreed"].append({"analysis_id": f.analysis_id, "source": f.source or "web",
                                     "comment": f.comment})
    return out
