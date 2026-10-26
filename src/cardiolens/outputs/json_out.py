"""Structured JSON output (TZ §11.1).

Includes measurements, findings, causal graph, confidences, critical flags,
model/rule-base versions, operating mode, and the decision/audit log — the
traceability backbone for regulator-readiness (TZ §1, §11, §12).
"""

from __future__ import annotations

import json
import math
from typing import Any

from cardiolens.pipeline import PipelineResult


def finite(obj: Any) -> Any:
    """``obj`` with every NaN or infinite float replaced by None (D165): JSON has no NaN,
    and a measurement that is not a number is "not measured", never a crash."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def result_to_dict(result: PipelineResult) -> dict[str, Any]:
    cfg = result.config
    r = result.report
    return finite({
        "versions": {
            "schema": cfg.versions.schema_version,
            "rules_base": cfg.versions.rules_base,
        },
        "mode": r.mode,
        "escalation_reason": r.escalation_reason,
        "patient": result.signal.patient.model_dump(),
        "context": result.signal.context.model_dump(),
        "quality": result.signal.quality.model_dump(),
        # TZ §12 (ECG scanner): interpretation layers and reports see the photo origin.
        "digitized_from_image": result.signal.provenance.digitized,
        "digitization_confidence": result.signal.provenance.confidence,
        "measurements": result.measurements.model_dump(),
        "findings": [f.model_dump() for f in result.findings.findings],
        "causal_graph": result.causal_graph.model_dump(by_alias=True),
        "critical_flags": result.causal_graph.critical_flags,
        "critical_alerts": [
            {"code": a.code, "label_ru": a.label_ru, "label_uz": a.label_uz,
             "reason": a.reason, "log_id": a.log_id, "unverified": a.unverified}
            for a in result.safety.alerts
        ],
        "serial_comparison": result.serial.to_dict() if result.serial else None,
        "interpretation": {
            "summary": [c.model_dump() for c in r.summary],
            "causal_explanation": [c.model_dump() for c in r.causal_explanation],
            "differential": [d.model_dump() for d in r.differential],
            "red_flags": [c.model_dump() for c in r.red_flags],
            "confidence": {
                "value": r.confidence_value,
                "word_ru": r.confidence_word_ru,
                "word_uz": r.confidence_word_uz,
            },
            "next_step": {"ru": r.next_step_ru, "uz": r.next_step_uz},
            "reasoning_trace": r.reasoning_trace,
            "provider": r.provider,
            "provider_model": r.provider_model,
            "disclaimer": {"ru": r.disclaimer_ru, "uz": r.disclaimer_uz},
        },
        "audit_log": result.audit_log,
    })


def write_json(result: PipelineResult, path: str) -> str:
    data = result_to_dict(result)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path
