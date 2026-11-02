"""Physician feedback capture and calibration dataset (TZ §13)."""

from cardiolens.feedback import quality
from cardiolens.feedback.store import (
    Feedback,
    FeedbackStore,
    FindingFeedback,
)

__all__ = ["Feedback", "FeedbackStore", "FindingFeedback", "quality"]

# Default location for the feedback log (overridable via env in the API).
DEFAULT_FEEDBACK_PATH = "data/feedback/feedback.jsonl"
DEFAULT_QUALITY_PATH = "data/feedback/quality.jsonl"   # D190: derived numbers per analysis
