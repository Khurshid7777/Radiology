"""Unified signal representation — the contract between all layers (TZ §3).

Every ingestion parser (Layer 1) MUST produce a ``SignalRepresentation`` so that
downstream layers never depend on the source format. Signal amplitude unit is
microvolts (µV), per TZ §4.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Vendors and datasets spell the augmented leads differently (PTB-XL WFDB headers
# say AVR/AVL/AVF; some exports use MLII, ECG_I, "Lead V1"). Everything downstream
# keys on the canonical spellings above, so parsers normalise names on ingestion.
_LEAD_ALIASES = {
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "VR": "aVR", "VL": "aVL", "VF": "aVF",
    "MLII": "II", "MLI": "I", "MLIII": "III",
    "LEAD1": "I", "LEAD2": "II", "LEAD3": "III",
}


def canonical_lead_name(name: str) -> str:
    """Map a raw lead label to the canonical spelling when it is a standard lead."""
    raw = str(name).strip()
    key = raw.upper().replace(" ", "").replace("_", "").replace("-", "")
    for prefix in ("LEAD", "ECG"):
        if key.startswith(prefix) and len(key) > len(prefix):
            key = key[len(prefix):]
    if key in _LEAD_ALIASES:
        return _LEAD_ALIASES[key]
    for std in STANDARD_12_LEADS:
        if key == std.upper():
            return std
    return raw


class PatientContext(BaseModel):
    """Optional clinical context feeding the causal analysis (TZ §3, §8)."""

    potassium: float | None = None       # mmol/L
    calcium: float | None = None         # mmol/L
    medications: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    prior_ecg_id: str | None = None


class Patient(BaseModel):
    """Demographics; age drives age-specific norms (TZ §1, §6)."""

    age_years: float | None = None
    age_days: float | None = None        # for neonates (TZ §1 "новорождённые")
    sex: str | None = None               # "male" | "female" | None


class Quality(BaseModel):
    """Per-lead signal quality (TZ §5)."""

    per_lead_sqi: list[float] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Where the signal came from and how trustworthy the digitisation is (TZ §4)."""

    digitized: bool = False              # True if recovered from image/scan
    confidence: float | None = None      # 0..1, low for dirty photos
    # For a digitised 3x4 page each lead only exists inside its 2.5-s time slot
    # (the rhythm strip covers the whole record). Per-lead measurements use only
    # the beats inside the window; outside it the samples are zero, not data.
    lead_windows: dict[str, tuple[float, float]] | None = None   # lead -> (t0_s, t1_s)
    # Times (s) where the record jumps between pages recorded at different moments
    # (two photos of a device's screen, D123): no RR interval spans one.
    discontinuities_s: list[float] | None = None
    notes: list[str] = Field(default_factory=list)


class SignalRepresentation(BaseModel):
    """Unified internal representation of an ECG (TZ §3).

    ``signal`` is a 2-D array shaped ``[n_leads, n_samples]`` in microvolts.
    It is stored as a numpy array; JSON export converts it to nested lists.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: np.ndarray = Field(..., description="float32[n_leads, n_samples], µV")
    sampling_rate_hz: float
    lead_names: list[str]
    duration_s: float
    source: str                          # wfdb|edf|dicom|scp|muse_xml|philips_xml|image|stream|csv|synthetic
    patient: Patient = Field(default_factory=Patient)
    context: PatientContext = Field(default_factory=PatientContext)
    quality: Quality = Field(default_factory=Quality)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("signal", mode="before")
    @classmethod
    def _coerce_signal(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("signal must be 2-D [n_leads, n_samples]")
        return arr

    @model_validator(mode="after")
    def _check_shapes(self) -> SignalRepresentation:
        n_leads = self.signal.shape[0]
        if len(self.lead_names) != n_leads:
            raise ValueError(
                f"lead_names ({len(self.lead_names)}) != signal leads ({n_leads})"
            )
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        return self

    @property
    def n_leads(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[1])

    def lead(self, name: str) -> np.ndarray:
        """Return the 1-D signal for a named lead."""
        return self.signal[self.lead_names.index(name)]

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (signal as nested lists)."""
        d = self.model_dump(exclude={"signal"})
        d["signal"] = self.signal.tolist()
        return d
