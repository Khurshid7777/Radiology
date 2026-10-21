"""Layer 6 — interpretation & explanation (TZ §9).

``interpret`` orchestrates: pick a provider (Claude if available, else stub) ->
generate raw report -> validate grounding (drop ungrounded claims) -> apply the
A/B/escalate mode policy.
"""

from __future__ import annotations

from cardiolens.contracts.report import InterpretationReport
from cardiolens.layer6_interpret.claude import ClaudeProvider, available
from cardiolens.layer6_interpret.grounding import validate_grounding
from cardiolens.layer6_interpret.local import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LocalOpenAIProvider,
    local_available,
)
from cardiolens.layer6_interpret.policy import decide_mode
from cardiolens.layer6_interpret.provider import InterpretationInput, LLMProvider
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.safety.critical import SafetyResult


def default_provider(prefer_claude: bool = True, llm=None) -> LLMProvider:
    """Pick the narrative provider from the ``llm`` config (or auto-detect)."""
    choice = (getattr(llm, "provider", "auto") or "auto").lower()
    base_url = getattr(llm, "local_base_url", DEFAULT_BASE_URL)
    model = getattr(llm, "local_model", DEFAULT_MODEL)
    timeout = float(getattr(llm, "timeout_s", 120.0))
    if choice == "stub":
        return StubProvider()
    if choice == "claude":
        return ClaudeProvider() if available() else StubProvider()
    if choice == "local":
        return LocalOpenAIProvider(base_url, model, timeout)
    # auto
    if prefer_claude and available():
        return ClaudeProvider()
    if local_available(base_url):
        return LocalOpenAIProvider(base_url, model, timeout)
    return StubProvider()


def provider_status(llm=None) -> dict:
    """What Layer 6 would use right now — for ``/health`` and the UI badge (TZ §12).

    Reports the configured choice, whether the local server answers, whether a
    Claude key+SDK are present, and the provider ``default_provider`` resolves to.
    """
    choice = (getattr(llm, "provider", "auto") or "auto").lower()
    base_url = getattr(llm, "local_base_url", DEFAULT_BASE_URL)
    model = getattr(llm, "local_model", DEFAULT_MODEL)
    local_up = local_available(base_url)
    active = default_provider(llm=llm)
    return {
        "configured": choice,
        "active": active.name,
        "model": getattr(active, "model", None) if active.name != "stub" else None,
        "local_base_url": base_url,
        "local_model": model,
        "local_server_up": local_up,
        "claude_available": available(),
    }


def interpret(
    data: InterpretationInput,
    safety: SafetyResult,
    *,
    provider: LLMProvider | None = None,
    min_overall: float,
    quality_ok: bool,
    digitized: bool = False,
) -> InterpretationReport:
    provider = provider or default_provider()
    report = provider.generate(data)

    validate_grounding(report, data)

    decision = decide_mode(
        data.mode, data.causal_graph, safety,
        min_overall=min_overall, quality_ok=quality_ok, digitized=digitized,
    )
    report.mode = decision.mode
    if decision.mode == "escalate":
        report.escalation_reason = decision.reason
        report.reasoning_trace.append(f"MODE B gate failed -> escalate: {decision.reason}")
    return report


__all__ = [
    "ClaudeProvider",
    "InterpretationInput",
    "LLMProvider",
    "LocalOpenAIProvider",
    "StubProvider",
    "default_provider",
    "interpret",
    "provider_status",
]
