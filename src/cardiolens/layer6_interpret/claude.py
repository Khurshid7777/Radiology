"""Anthropic Claude narrative provider for Layer 6 (TZ §9, §12).

Sends the causal graph + findings + measurements to Claude and requests a
schema-constrained bilingual (RU/UZ) interpretation. The model is instructed to
ground every claim in a provided token; the grounding validator enforces this
afterwards regardless of what the model returns (defence in depth, TZ §9).

PHI note (TZ §12): only derived features/labels are sent, never raw identifiers.
Requires ``ANTHROPIC_API_KEY`` and the optional ``anthropic`` dependency; if
absent, callers should fall back to ``StubProvider``.
"""

from __future__ import annotations

import json
import os

from cardiolens.contracts.report import Claim, InterpretationReport
from cardiolens.layer6_interpret.provider import InterpretationInput, LLMProvider
from cardiolens.layer6_interpret.stub import StubProvider

_SYSTEM = (
    "You are a cardiology ECG interpretation assistant. You are given structured "
    "findings and a causal graph derived by a deterministic engine. Write a "
    "bilingual (Russian + Uzbek) interpretation. CRITICAL RULES: (1) Do NOT invent "
    "findings or causal links absent from the provided data. (2) Every claim must "
    "cite a grounding token from the provided list in 'grounded_by'. (3) Output "
    "ONLY valid JSON matching the requested schema. (4) Provide a visible "
    "reasoning trace."
)


def available() -> bool:
    """True if a Claude call is possible in this environment."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        # Model is configurable; defaults to a current Claude model, overridable
        # via the ANTHROPIC_MODEL environment variable.
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.max_tokens = max_tokens

    def _payload(self, data: InterpretationInput) -> dict:
        return {
            "mode": data.mode,
            "grounding_tokens": sorted(data.grounding_tokens()),
            "findings": [
                {"code": f.code, "ru": f.label_ru, "uz": f.label_uz,
                 "confidence": f.confidence,
                 "evidence": [f"{e.feature}={e.value}" for e in f.evidence]}
                for f in data.findings.findings
            ],
            "causal_edges": [
                {"id": f"{e.from_}->{e.to}", "relation": e.relation}
                for e in data.causal_graph.edges
            ],
            "critical_flags": data.causal_graph.critical_flags,
            "implausible_flags": data.causal_graph.implausible_flags,
            "confidence": data.causal_graph.confidence,
            "measurements": _measurements(data),
        }

    def generate(self, data: InterpretationInput) -> InterpretationReport:
        prompt = PROMPT_INSTRUCTIONS + json.dumps(self._payload(data), ensure_ascii=False, indent=2)
        # The narrative is the last layer: a refused key, an unpaid account, an outage or
        # malformed JSON must not fail the whole analysis (D154) — same as the local provider.
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            report = report_from_json(json.loads(_extract_json(text)), data)
            report.provider, report.provider_model = self.name, self.model
            report.reasoning_trace.insert(0, f"narrative by Claude {self.model}")
            return report
        except Exception as e:
            report = StubProvider().generate(data)   # provider stays "stub": the UI must not claim a model
            report.reasoning_trace.insert(
                0, f"Claude unavailable ({type(e).__name__}) -> deterministic offline narrative")
            return report


# Findings that mean the numbers were not read from a trustworthy signal (D128): the
# model gets none of them, or a false rate from a stretched photo would be narrated.
WITHHOLD_MEASUREMENTS = ("photo_unreliable", "no_ecg_signal", "amplitude_implausible")


def _measurements(data: InterpretationInput) -> dict:
    codes = {f.code for f in data.findings.findings}
    withheld = [c for c in WITHHOLD_MEASUREMENTS if c in codes]
    if withheld:
        return {"withheld": f"not measured reliably ({', '.join(withheld)}); do not state any rate or interval"}
    m = data.measurements
    return {
        "heart_rate_bpm": m.heart_rate_bpm,
        "pr_ms": m.intervals.pr_ms,
        "qrs_ms": m.intervals.qrs_ms,
        "qtc_bazett_ms": m.intervals.qtc_bazett_ms,
        "qtc_fridericia_ms": m.intervals.qtc_fridericia_ms,
    }


PROMPT_INSTRUCTIONS = (
    "Interpret this ECG. Return JSON with keys: summary[], "
    "causal_explanation[], differential[], red_flags[], "
    "confidence_word_ru, confidence_word_uz, next_step_ru, next_step_uz, "
    "reasoning_trace[]. Each summary/causal/red_flag item is "
    "{text_ru, text_uz, grounded_by[]}.\n\nDATA:\n"
)


def report_from_json(parsed: dict, data: InterpretationInput) -> InterpretationReport:
    """Assemble the raw report from a model's JSON (shared by cloud/local providers)."""

    def claims(key: str) -> list[Claim]:
        return [Claim(**c) for c in parsed.get(key, [])]

    return InterpretationReport(
        mode=data.mode,
        summary=claims("summary"),
        causal_explanation=claims("causal_explanation"),
        red_flags=claims("red_flags"),
        differential=[],  # kept deterministic from the engine
        confidence_value=data.causal_graph.confidence,
        confidence_word_ru=parsed.get("confidence_word_ru", ""),
        confidence_word_uz=parsed.get("confidence_word_uz", ""),
        next_step_ru=parsed.get("next_step_ru", ""),
        next_step_uz=parsed.get("next_step_uz", ""),
        reasoning_trace=[str(t) for t in parsed.get("reasoning_trace", [])],
        disclaimer_ru=data.disclaimer_ru,
        disclaimer_uz=data.disclaimer_uz,
    )


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return text[start : end + 1]
