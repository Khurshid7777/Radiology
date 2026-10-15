"""Configuration loading and validation (TZ §0 — nothing hard-coded).

All runtime thresholds, norms and the pathophysiology rule base live in
``configs/`` and are loaded through here. Norm files are validated loosely (they
are reference tables); the rule base is validated against ``RuleBase``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from cardiolens.contracts.causal import RuleBase


def _repo_root() -> Path:
    # src/cardiolens/config.py -> repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return _repo_root() / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Disclaimers(BaseModel):
    ru: str
    uz: str


class ConfidenceGates(BaseModel):
    autonomous_min_overall: float
    autonomous_min_component: float


class Preprocessing(BaseModel):
    target_sampling_rate_hz: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    sqi_accept_threshold: float


class Versions(BaseModel):
    schema_version: str = Field(alias="schema")
    rules_base: str

    model_config = {"populate_by_name": True}


class LLMConfig(BaseModel):
    """Layer-6 narrative provider selection (TZ §12): auto | claude | local | stub."""

    provider: str = "auto"
    local_base_url: str = "http://127.0.0.1:11434/v1"
    local_model: str = "llama3.1:8b"
    timeout_s: float = 120.0


class MLConfig(BaseModel):
    """Learnable layer (TZ §17): off, or a joblib artifact used as extra evidence."""

    enabled: bool = True
    model_path: str = "configs/models/rhythm_hgb.joblib"   # relative to configs/ parent


class PauseConfig(BaseModel):
    """Pause / dropped-beat thresholds (TZ §7.1, §10)."""
    finding_ms: float = 2000.0
    ratio: float = 1.8
    critical_ms: float = 3000.0


class PacingConfig(BaseModel):
    """Paced-rhythm spike detection (TZ §7.1): an extreme slope outlier time-locked to
    the QRS, or a spike-shaped event seen by many leads at once (D74-D75)."""
    spike_ratio_min: float = 5.5
    coupling_min: float = 0.2
    multilead_k: float = 3.0
    multilead_min_leads: int = 6
    multilead_coupling_min: float = 0.3


class QualityConfig(BaseModel):
    """Signal-organization gate (TZ §5): below beat_consistency_min the QRS complexes
    are not self-consistent (artifact/noise), so R-peak-derived findings are withheld."""
    beat_consistency_min: float = 0.8
    # ... unless nearly every beat has a near-identical twin on the strip (D89): two or
    # more stable QRS shapes (frequent PVCs, bigeminy, intermittent pacing) blend the
    # single median template, but noise does not repeat itself.
    beat_partner_min: float = 0.95


class AppConfig(BaseModel):
    mode: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    pause: PauseConfig = Field(default_factory=PauseConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    mains_frequency_hz: float
    preprocessing: Preprocessing
    confidence: ConfidenceGates
    disclaimers: Disclaimers
    versions: Versions


# Environment overrides for the deployment-specific LLM entries of app.yaml, so a
# container or a shell can point Layer 6 at its own server without editing the
# repo (docker-compose: CARDIOLENS_LLM_BASE_URL=http://ollama:11434/v1).
_LLM_ENV = {
    "CARDIOLENS_LLM_PROVIDER": "provider",
    "CARDIOLENS_LLM_BASE_URL": "local_base_url",
    "CARDIOLENS_LLM_MODEL": "local_model",
    "CARDIOLENS_LLM_TIMEOUT_S": "timeout_s",
}


def _apply_llm_env(raw: dict[str, Any]) -> dict[str, Any]:
    llm = dict(raw.get("llm") or {})
    for var, key in _LLM_ENV.items():
        value = os.environ.get(var)
        if value:
            llm[key] = value
    raw["llm"] = llm
    return raw


@functools.lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    return AppConfig(**_apply_llm_env(_load_yaml(configs_dir() / "app.yaml")))


@functools.lru_cache(maxsize=1)
def load_rule_base() -> RuleBase:
    return RuleBase(**_load_yaml(configs_dir() / "rules" / "pathophysiology.yaml"))


@functools.lru_cache(maxsize=2)
def load_risk_tiers() -> dict[str, Any]:
    """Clinical risk tiers of finding codes and the protocol's targets (TZ §16)."""
    return _load_yaml(configs_dir() / "risk_tiers.yaml")


def load_norms(group: str) -> dict[str, Any]:
    """Load an age-group norm table ('adult' | 'pediatric')."""
    return _load_yaml(configs_dir() / "norms" / f"{group}.yaml")


def norms_for_patient(age_years: float | None, age_days: float | None) -> dict[str, Any]:
    """Pick the appropriate norm table for a patient (TZ §6).

    Children (age < 16y, or age given only in days) use pediatric norms.
    """
    is_pediatric = False
    if age_days is not None and age_years is None or age_years is not None and age_years < 16:
        is_pediatric = True
    return load_norms("pediatric" if is_pediatric else "adult")
