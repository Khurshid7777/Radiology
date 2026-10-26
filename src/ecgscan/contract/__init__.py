"""Scanner output contract and its JSON Schema."""

from pathlib import Path

from ecgscan.contract.models import (
    DIGITIZED_FLAG,
    REJECT_CODES,
    STANDARD_LEADS,
    WARNING_CODES,
    Calibration,
    Candidate,
    Capture,
    CaptureCorners,
    CaptureQC,
    Confirmation,
    Layout,
    QualitySummary,
    ScanResult,
    Source,
    Trace,
)

SCHEMA_PATH = Path(__file__).with_name("scan_result.schema.json")

__all__ = [
    "DIGITIZED_FLAG",
    "REJECT_CODES",
    "SCHEMA_PATH",
    "STANDARD_LEADS",
    "WARNING_CODES",
    "Calibration",
    "Candidate",
    "Capture",
    "CaptureCorners",
    "CaptureQC",
    "Confirmation",
    "Layout",
    "QualitySummary",
    "ScanResult",
    "Source",
    "Trace",
]
