"""Mode policy for autonomous vs decision-support output (TZ §2, §9, §10).

Mode A (decision support) always emits a report for the physician. Mode B
(autonomous) emits an assertive report ONLY when every gate passes; otherwise it
escalates to "требуется просмотр врача" with a reason. A critical finding never
permits autonomous output (TZ §10).
"""

from __future__ import annotations

from dataclasses import dataclass

from cardiolens.contracts.causal import CausalGraph
from cardiolens.safety.critical import SafetyResult


@dataclass
class ModeDecision:
    mode: str                 # "A" | "B" | "escalate"
    reason: str | None = None


def decide_mode(
    requested_mode: str,
    causal_graph: CausalGraph,
    safety: SafetyResult,
    *,
    min_overall: float,
    quality_ok: bool,
    digitized: bool = False,
) -> ModeDecision:
    """Apply the §2 gating logic."""
    if requested_mode != "B":
        return ModeDecision(mode="A")

    # Mode B requested — check every gate (TZ §2).
    if safety.has_critical or causal_graph.critical_flags:
        return ModeDecision(
            mode="escalate",
            reason="критическая находка требует просмотра врача (§10)",
        )
    # A signal digitised from a photo carries distortions the measurements cannot see
    # (ECG scanner SD-015): it is always read by a doctor.
    if digitized:
        return ModeDecision(
            mode="escalate",
            reason="сигнал получен по фото — автономная интерпретация не применяется",
        )
    if causal_graph.implausible_flags:
        return ModeDecision(
            mode="escalate",
            reason="физиологически неправдоподобные находки — возможен артефакт/"
                   "перестановка электродов (§8.3)",
        )
    if not quality_ok:
        return ModeDecision(
            mode="escalate",
            reason="качество сигнала ниже порога (§2)",
        )
    if causal_graph.confidence < min_overall:
        return ModeDecision(
            mode="escalate",
            reason=f"общая уверенность {causal_graph.confidence} < порога {min_overall} (§2)",
        )
    return ModeDecision(mode="B")
