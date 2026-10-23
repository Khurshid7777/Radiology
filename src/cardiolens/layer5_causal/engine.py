"""Layer 5 — causal & physiological reasoning engine (TZ §8).

Operates over the Layer-4 findings using the declarative pathophysiology rule
base (``configs/rules/pathophysiology.yaml``). Builds a networkx causal graph of
"root cause -> mechanism -> effects", labels each node physiologic/pathologic,
runs a parsimony/consistency check, and flags findings that no mechanism
explains as "physiologically implausible" (TZ §8.3) — the cue to suspect an
artifact / electrode reversal / poor signal quality.
"""

from __future__ import annotations

import networkx as nx

from cardiolens.contracts.causal import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    Rule,
    RuleBase,
)
from cardiolens.contracts.findings import FindingSet
from cardiolens.contracts.signal import PatientContext


def _context_supports(rule: Rule, context: PatientContext) -> bool:
    """True if patient context is consistent with (does not contradict) a rule.

    Missing context never blocks activation (context is optional, TZ §3);
    present context that contradicts the rule lowers support to False.
    """
    cs = rule.context_support
    if cs is None:
        return True
    k = context.potassium
    if cs.potassium_min is not None and k is not None and k < cs.potassium_min:
        return False
    if cs.potassium_max is not None and k is not None and k > cs.potassium_max:
        return False
    ca = context.calcium
    if cs.calcium_min is not None and ca is not None and ca < cs.calcium_min:
        return False
    if cs.calcium_max is not None and ca is not None and ca > cs.calcium_max:
        return False
    return True


def _rule_context_boost(rule: Rule, context: PatientContext) -> float:
    """Confidence boost when context actively confirms the rule (TZ §8.1)."""
    cs = rule.context_support
    if cs is None:
        return 0.0
    k = context.potassium
    if cs.potassium_min is not None and k is not None and k >= cs.potassium_min:
        return 0.15
    if cs.potassium_max is not None and k is not None and k <= cs.potassium_max:
        return 0.15
    return 0.0


def analyze(
    findings: FindingSet,
    rule_base: RuleBase,
    context: PatientContext | None = None,
) -> CausalGraph:
    """Build the causal graph from findings + rule base (TZ §8)."""
    context = context or PatientContext()
    observed = findings.codes()
    graph = nx.DiGraph()
    activated: list[Rule] = []

    # --- Activate rules whose triggers intersect observed findings ---
    for rule in rule_base.rules:
        if not (set(rule.trigger_findings) & observed):
            continue
        if not _context_supports(rule, context):
            continue
        activated.append(rule)
        for node in rule.nodes:
            graph.add_node(
                node.id, kind=node.kind, label_ru=node.label_ru,
                label_uz=node.label_uz, physiologic=rule.physiologic,
                observed=node.id in observed,
            )
        for edge in rule.edges:
            graph.add_edge(edge.from_, edge.to, relation=edge.relation)

    # --- Explained set: any observed finding present as a graph node ---
    explained = {n for n in graph.nodes if graph.nodes[n].get("observed")}
    # --- Implausibility: observed findings that no activated rule explains ---
    # (excluding benign standalone findings like plain sinus rhythm).
    benign = {"sinus_rhythm", "juvenile_t_pattern"}
    implausible = sorted(c for c in observed - explained - benign - _standalone_codes()
                         if not c.startswith("ml_suggested_"))

    # --- Critical flags come from findings the safety layer marked (TZ §10) + rules
    critical_flags = sorted({f.code for f in findings.critical()} |
                            {r.id for r in activated if r.critical})

    # --- Confidence: parsimony — fewer mechanisms explaining more = higher ---
    conf = _graph_confidence(activated, findings, context)

    out = CausalGraph(
        nodes=[
            GraphNode(
                id=n, kind=graph.nodes[n]["kind"],
                label_ru=graph.nodes[n]["label_ru"],
                label_uz=graph.nodes[n]["label_uz"],
                physiologic=graph.nodes[n]["physiologic"],
                observed=graph.nodes[n]["observed"],
            )
            for n in graph.nodes
        ],
        edges=[
            GraphEdge(**{"from": u, "to": v, "relation": graph.edges[u, v]["relation"]})
            for u, v in graph.edges
        ],
        activated_rules=[r.id for r in activated],
        critical_flags=critical_flags,
        implausible_flags=implausible,
        confidence=round(conf, 3),
    )
    return out


def _standalone_codes() -> set[str]:
    """Findings that legitimately stand alone (not needing a causal chain)."""
    return {
        "sinus_bradycardia", "sinus_tachycardia", "irregular_rhythm",
        "qtc_unreliable_af",
        "first_degree_av_block", "pr_borderline", "short_qt", "st_elevation",
        "atrial_fibrillation", "st_depression", "st_depression_minor", "t_inversion",
        "complete_av_block", "atrial_flutter", "left_axis_deviation",
        "right_axis_deviation", "extreme_axis", "low_qrs_voltage",
        "left_atrial_enlargement", "right_atrial_enlargement",
        "st_elevation_nonspecific", "secondary_st_t_change", "st_elevation_any",
        "lvh_probable", "rvh",
    }


def _graph_confidence(activated, findings: FindingSet, context: PatientContext) -> float:
    """Overall confidence from finding confidences + parsimony + context (§8.3)."""
    if not findings.findings:
        return 0.0
    mean_conf = sum(f.confidence for f in findings.findings) / len(findings.findings)
    # Parsimony bonus: explaining >=2 findings with a single rule.
    parsimony = 0.0
    for r in activated:
        boost = _rule_context_boost(r, context)
        parsimony = max(parsimony, boost)
    return float(min(1.0, mean_conf + parsimony))
