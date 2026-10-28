"""``ScanResult`` — the scanner's output contract (docs/scanner/REPO_AUDIT.md §9).

Mirrors ``scan_result.schema.json`` (shipped next to this file); a test keeps the
two in agreement. Amplitudes are mV, time is seconds, a missing sample is ``None``
in JSON and NaN in numpy (SD-006, SD-007).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ecgscan import PIPELINE_VERSION, SCHEMA_VERSION

Status = Literal["ok", "needs_confirmation", "rejected"]
CalSource = Literal["calibration_pulse", "header_ocr", "format_registry", "user"] | None
TraceMethod = Literal["dp_path", "legacy_track", "hybrid", "weighted_centroid", "column_median", "onnx"]

# Trace flag: a rhythm strip unrelated to every panel recorded with it (SD-060).
RHYTHM_MISMATCH = "RHYTHM_MISMATCH"

REJECT_CODES = (
    "LOW_RESOLUTION", "GLARE", "BLUR", "GRID_NOT_FOUND", "CALIBRATION_MISSING", "LAYOUT_UNKNOWN",
    "TRACE_BROKEN", "LIMB_LEADS_INCONSISTENT", "LOW_CONTRAST", "SCREEN_PHOTO", "NOT_AN_ECG",
    "PAGE_INCOMPLETE", "PROCESSING_ERROR", "IMAGE_TOO_LARGE", "BOARD_NOT_FOUND",
)
WARNING_CODES = (
    "SPEED_UNCERTAIN", "GAIN_UNCERTAIN", "SPEED_SOURCES_DISAGREE", "GAIN_SOURCES_DISAGREE",
    "LAYOUT_UNCERTAIN", "LABEL_ORDER_CONFLICT", "LIMB_LEADS_INCONSISTENT", "TRACE_GAPS",
    "LEAD_INCOMPLETE", "BEATS_INCONSISTENT", "QRS_UNSTABLE", "RHYTHM_INCONSISTENT", "LOW_LIGHT", "LEAD_OVERLAP", "LOW_RESOLUTION_MARGINAL", "FADED_PAPER", "STITCHED",
    "LOW_RESOLUTION_COARSE", "PAGES_STITCHED", "SEAM_UNCERTAIN",
)
DIGITIZED_FLAG = "digitized_from_image"
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

_Unit = Field(ge=0.0, le=1.0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaptureQC(_Strict):
    """The phone's frame checks at the moment of capture (TZ addendum §18.13, SD-071).
    Same names and meaning as ``ecgscan.capture_qc.FrameQC``; ``failed_checks`` = its hints."""
    brightness: float | None = Field(default=None, ge=0, le=255)
    glare: float | None = Field(default=None, ge=0, le=1)
    sharpness: float | None = Field(default=None, ge=0)
    shadow: float | None = Field(default=None, ge=0, le=1)
    px_per_mm: float | None = Field(default=None, gt=0)
    tilt_deg: float | None = None
    sheet_area: float | None = Field(default=None, ge=0, le=1)
    passed: bool
    failed_checks: list[str] = []


class CaptureCorners(_Strict):
    auto: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)
    final: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)
    edited_by_user: bool = False


class Capture(_Strict):
    """How the photo was taken (TZ addendum §18.13). Device model, resolution and the
    masking flag are in ``Source`` already (``device``, ``width_px``/``height_px``,
    ``header_masked``) and are not repeated; ``corners.final`` is also ``client_crop``."""
    capture_mode: Literal["custom_camera", "gallery"] = "custom_camera"
    auto_capture: bool = False
    corners: CaptureCorners | None = None
    camera_tilt_deg: float | None = Field(default=None, ge=0, le=90)   # gyroscope: phone vs the sheet's plane
    # The doctor's quarter turn on the crop screen, clockwise: applied by the server before
    # reading, the photo itself is never re-encoded (§18.1). Corners are before the turn.
    rotate_deg: Literal[0, 90, 180, 270] = 0
    qc: CaptureQC | None = None
    mask_method: Literal["template_zone", "profile_zones", "manual", "none"] | None = None


class Source(_Strict):
    modality: Literal["photo", "scan", "pdf"] = "photo"
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_bytes: int | None = Field(default=None, ge=0)
    mime_type: str | None = None
    width_px: int | None = Field(default=None, ge=1)
    height_px: int | None = Field(default=None, ge=1)
    exif_orientation: int | None = Field(default=None, ge=1, le=8)
    captured_at: str | None = None
    received_at: str | None = None
    device: str | None = None
    client: Literal["flutter", "telegram_bot", "api", "web", "cli"] | None = None
    client_crop: list[list[float]] | None = None
    header_masked: bool = False
    # A page photographed in parts on a capture board (SD-066): every part's hash, in the
    # order sent; original_sha256 is then the hash of their concatenated hex digests.
    parts_sha256: list[str] | None = None
    capture: Capture | None = None
    pipeline_version: str = PIPELINE_VERSION


class Candidate(_Strict):
    value: float
    source: Literal["calibration_pulse", "header_ocr", "format_registry"]
    confidence: float = _Unit


class Calibration(_Strict):
    px_per_mm_x: float | None = Field(default=None, gt=0)
    px_per_mm_y: float | None = Field(default=None, gt=0)
    grid_residual_mm: float | None = Field(default=None, ge=0)
    paper_speed_mm_s: Literal[25, 50] | None = None
    gain_mm_mv: Literal[5, 10, 20] | None = None
    speed_source: CalSource = None
    gain_source: CalSource = None
    speed_candidates: list[Candidate] = Field(default_factory=list)
    gain_candidates: list[Candidate] = Field(default_factory=list)
    confidence: float = _Unit


class Layout(_Strict):
    name: str | None = None
    registry_version: str | None = None
    confidence: float = _Unit
    label_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmed_by_user: bool = False          # the layout came from the doctor (a hint or a correction), S2
    device_profile_id: str | None = None     # configs/scanner/devices/<id>.yaml the layout came from, S2


class Trace(_Strict):
    lead: str | None = None
    samples_mv: list[float | None]
    start_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)
    quality: float = _Unit
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    method: TraceMethod | None = None
    box_px: list[int] | None = None
    flags: list[str] = Field(default_factory=list)

    @classmethod
    def from_array(cls, samples_mv: np.ndarray, *, start_s: float, fs: float, **kw) -> Trace:
        arr = np.asarray(samples_mv, dtype=np.float64)
        missing = ~np.isfinite(arr)
        kw.setdefault("missing_fraction", round(float(missing.mean()), 4) if arr.size else 1.0)
        return cls(samples_mv=[None if m else round(float(v), 5) for v, m in zip(arr, missing, strict=True)],
                   start_s=float(start_s), duration_s=float(len(arr) / fs), **kw)

    def array(self) -> np.ndarray:
        """Samples as float64 with NaN for missing."""
        return np.array([np.nan if v is None else v for v in self.samples_mv], dtype=np.float64)


class QualitySummary(_Strict):
    overall: float = _Unit
    limb_lead_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    px_per_mm: float | None = Field(default=None, ge=0.0)
    sharpness: float | None = None
    glare_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    trace_contrast: float | None = None
    processing_ms: int | None = Field(default=None, ge=0)


class Confirmation(_Strict):
    by: str
    at: str
    decision: Literal["accepted", "corrected", "retake"]
    changes: dict | None = None


class Page(_Strict):
    """One sheet of a multi-page study (TZ 18.4, S2), in the order the doctor gave."""

    index: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Status
    layout: str | None = None
    overlap_with_next: bool | None = None    # the next sheet repeats this one's last seconds (found, not assumed)
    overlap_s: float | None = Field(default=None, ge=0.0)   # ... by this much


class ScanResult(_Strict):
    schema_version: str = SCHEMA_VERSION
    scan_id: str
    status: Status
    reject_reasons: list[Literal[REJECT_CODES]] = Field(default_factory=list)  # type: ignore[valid-type]
    warnings: list[Literal[WARNING_CODES]] = Field(default_factory=list)  # type: ignore[valid-type]
    source: Source
    calibration: Calibration
    layout: Layout
    sampling_rate_hz: float = Field(gt=0)
    leads: dict[str, Trace] = Field(default_factory=dict)
    rhythm: list[Trace] = Field(default_factory=list)
    quality: QualitySummary
    artifacts: dict[str, str] = Field(default_factory=dict)
    provenance_flags: list[str] = Field(default_factory=lambda: [DIGITIZED_FLAG])
    confirmation: Confirmation | None = None
    pages: list[Page] | None = None          # a multi-page study (S2): the sheets, stitched in this order

    @model_validator(mode="after")
    def _rules(self) -> ScanResult:
        if (self.status == "rejected") != bool(self.reject_reasons):
            raise ValueError("reject_reasons must be non-empty exactly when status == 'rejected'")
        if self.status == "ok" and (self.calibration.paper_speed_mm_s is None or self.calibration.gain_mm_mv is None):
            raise ValueError("status 'ok' requires an established paper speed and gain")
        if DIGITIZED_FLAG not in self.provenance_flags:
            raise ValueError(f"provenance_flags must contain '{DIGITIZED_FLAG}'")
        bad = set(self.leads) - set(STANDARD_LEADS)
        if bad:
            raise ValueError(f"non-canonical lead names: {sorted(bad)}")
        for t in self.rhythm:
            if not t.lead:
                raise ValueError("rhythm traces must name their lead")
        return self

    def all_traces(self) -> list[tuple[str, Trace]]:
        """(lead, trace) for panels and rhythm strips."""
        return [(k, t) for k, t in self.leads.items()] + [(t.lead or "", t) for t in self.rhythm]

    def to_json_dict(self) -> dict:
        d = self.model_dump(mode="json")
        # Rounding keeps payloads small; NaN never reaches JSON (None instead).
        for t in list(d["leads"].values()) + d["rhythm"]:
            t["samples_mv"] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
                               for v in t["samples_mv"]]
        return d
