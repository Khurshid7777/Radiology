"""Deterministic offline interpretation provider (TZ §9, §12).

Builds a fully-grounded ``InterpretationReport`` directly from the findings and
causal graph — no external model, no network. Used for tests, CI, and
full-offline operation. Every claim cites a real evidence feature or graph edge,
so it passes the grounding validator by construction.
"""

from __future__ import annotations

from cardiolens.contracts.report import (
    Claim,
    Differential,
    InterpretationReport,
)
from cardiolens.layer6_interpret.provider import InterpretationInput, LLMProvider


def _conf_word(c: float) -> tuple[str, str]:
    if c >= 0.85:
        return "высокая", "yuqori"
    if c >= 0.6:
        return "умеренная", "o'rtacha"
    return "низкая", "past"


RELATION_WORDS = {
    "causes": ("причина", "sabab"),
    "progresses_to": ("ведёт к", "olib keladi"),
    "reciprocal": ("реципрокно", "resiprok"),
    "explained_by": ("объясняется", "izohlanadi"),
}


class StubProvider(LLMProvider):
    name = "stub"

    def generate(self, data: InterpretationInput) -> InterpretationReport:
        trace: list[str] = []
        summary: list[Claim] = []
        for f in data.findings.findings:
            grounds = [f.code] + [e.feature for e in f.evidence]
            summary.append(Claim(text_ru=f.label_ru, text_uz=f.label_uz,
                                 grounded_by=grounds))
            trace.append(f"finding {f.code} (conf={f.confidence})")

        # Causal explanation from graph edges (TZ §9b). A rule carries all its branches (a
        # bundle-branch rule: RBBB and LBBB); an edge to or from a finding the record does not
        # show is not said (D174: "Блокада ножки → БПНПГ" was printed for an LBBB).
        causal: list[Claim] = []
        nodes = {n.id: n for n in data.causal_graph.nodes}
        for edge in data.causal_graph.edges:
            a, b = nodes.get(edge.from_), nodes.get(edge.to)
            if not (a and b) or any(n.kind == "finding" and not n.observed for n in (a, b)):
                continue
            edge_id = f"{edge.from_}->{edge.to}"
            rel_ru, rel_uz = RELATION_WORDS.get(edge.relation, (edge.relation, edge.relation))
            causal.append(Claim(
                text_ru=f"{a.label_ru} → {b.label_ru} ({rel_ru})",
                text_uz=f"{a.label_uz} → {b.label_uz} ({rel_uz})",
                grounded_by=[edge_id],
            ))
            trace.append(f"edge {edge_id} [{edge.relation}]")

        # Red flags from critical labels (TZ §9e, §10).
        red_flags = [
            Claim(text_ru=ru, text_uz=uz, grounded_by=list(data.causal_graph.critical_flags) or ["critical"])
            for ru, uz in data.critical_labels
        ]

        # Differential: rank pathologic findings by confidence.
        diff: list[Differential] = []
        for i, f in enumerate(sorted(data.findings.findings,
                                     key=lambda x: -x.confidence), start=1):
            diff.append(Differential(label_ru=f.label_ru, label_uz=f.label_uz,
                                     priority=i, confidence=f.confidence))

        conf = data.causal_graph.confidence
        cw_ru, cw_uz = _conf_word(conf)

        if data.causal_graph.implausible_flags:
            trace.append(
                "implausible findings present -> suspect artifact/electrode reversal"
            )
        next_ru = "Рассмотреть клиническую корреляцию и, при необходимости, повторную запись ЭКГ."
        next_uz = "Klinik korrelyatsiyani ko'rib chiqish va zarur bo'lsa EKGni qayta olish."

        return InterpretationReport(
            mode=data.mode,
            summary=summary,
            causal_explanation=causal,
            differential=diff,
            confidence_value=conf,
            confidence_word_ru=cw_ru,
            confidence_word_uz=cw_uz,
            red_flags=red_flags,
            next_step_ru=next_ru,
            next_step_uz=next_uz,
            reasoning_trace=trace,
            disclaimer_ru=data.disclaimer_ru,
            disclaimer_uz=data.disclaimer_uz,
        )
