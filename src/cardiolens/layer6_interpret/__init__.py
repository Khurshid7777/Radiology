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

GATED_CODES = ("photo_unreliable", "no_ecg_signal", "amplitude_implausible")     # as outputs.display.GATED
RETAKE_PAPER = (
    ("Переснимите ЭКГ: весь лист в кадре и ближе — клетки сетки должны быть чётко видны; ровно, без бликов и теней. "
    "Отправьте как файл (не сжатое фото) и повторите анализ."),
    ("EKGni qayta suratga oling: varaq kadrga to‘liq sig‘sin va yaqinroqdan — katak chiziqlari aniq ko‘rinsin; tekis, "
    "yaltirashsiz va soyasiz. Fayl sifatida yuboring (siqilgan rasm emas) va tahlilni takrorlang."))
RETAKE_SCREEN = (
    "Переснимите экран прибора: экран целиком в кадре, резко и без бликов; затем повторите анализ.",
    "Qurilma ekranini qayta suratga oling: ekran kadrga to‘liq sig‘sin, aniq va yaltirashsiz; so‘ng tahlilni takrorlang.")
RECHECK_RECORD = (
    "Сначала оцените пациента; затем проверьте запись (электроды, единицы и масштаб файла) и повторите анализ.",
    "Avval bemorni baholang; so‘ng yozuvni tekshiring (elektrodlar, fayl birliklari va masshtabi) va tahlilni takrorlang.")


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
    screen: bool = False,
) -> InterpretationReport:
    provider = provider or default_provider()
    # D343: a record under a reliability gate has nothing to interpret (measurements are withheld) — the narrative is the
    # deterministic one and the next step is fixed: a model's free text there printed internal field names and told the
    # owner of a paper strip to "fill the frame with the screen".
    gated = bool(data.findings.codes() & set(GATED_CODES))
    if gated:
        provider = StubProvider()
    report = provider.generate(data)
    if gated:
        report.next_step_ru, report.next_step_uz = (RETAKE_SCREEN if screen else RETAKE_PAPER) if digitized else RECHECK_RECORD

    validate_grounding(report, data)

    decision = decide_mode(
        data.mode, data.causal_graph, safety,
        min_overall=min_overall, quality_ok=quality_ok, digitized=digitized,
    )
    report.mode = decision.mode
    if decision.mode == "escalate":
        report.escalation_reason = decision.reason
        report.reasoning_trace.append(f"Автономный режим B не применён, нужен просмотр врача: {decision.reason}")
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
