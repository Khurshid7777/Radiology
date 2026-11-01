"""Findings contract — output of Layer 4 (TZ §7).

A finding is a bilingual, evidence-backed statement of *what* is present on the
ECG. *Why* and *whether it is coherent* is Layer 5's job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single feature->value pair backing a finding (TZ §7)."""

    feature: str
    value: str


class Finding(BaseModel):
    """SCP-like finding statement (TZ §7).

    ``code`` is a stable machine identifier (used to trigger causal rules and to
    match critical-finding detectors). Labels are bilingual per TZ §0.
    """

    code: str
    label_ru: str
    label_uz: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    # Group per TZ §7: rhythm | ischemia | interval | hypertrophy | other
    group: str = "other"
    critical: bool = False  # set by safety layer (TZ §10)

    def evidence_map(self) -> dict[str, str]:
        return {e.feature: e.value for e in self.evidence}


class FindingSet(BaseModel):
    """Collection of findings with quick lookup helpers."""

    findings: list[Finding] = Field(default_factory=list)

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}

    def by_code(self, code: str) -> Finding | None:
        return next((f for f in self.findings if f.code == code), None)

    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.critical]


# D189: a photo gated only because its limb leads disagree, read off a good lead-II strip,
# keeps its rate. The gate finding carries it as evidence; nothing else is released.
RATE_RELEASED = "rate_released_bpm"


def released_rate(fs: FindingSet) -> float | None:
    """The rate a gated photo still states (D189), or None when it states none."""
    f = fs.by_code("photo_unreliable")
    v = f.evidence_map().get(RATE_RELEASED) if f else None
    try:
        return float(v) if v else None
    except ValueError:
        return None
