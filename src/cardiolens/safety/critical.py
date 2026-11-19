"""Cross-cutting safety layer — critical "must-not-miss" findings (TZ §10).

These detectors raise a prominent alert INDEPENDENTLY of the LLM and of the
operating mode, and always appear in the report. Priority is sensitivity
(recall) over specificity — a false alarm beats a miss (TZ §10). In autonomous
mode B, any critical finding forces escalation to a physician (TZ §2, §10).
Every alert is logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import FindingSet

# Codes considered critical when present (mapped from Layer-4 findings).
# TZ §10 must-not-miss list; recall is prioritised over specificity.
CRITICAL_CODES = {
    "ventricular_fibrillation": ("Фибрилляция / трепетание желудочков (D204)",
                                 "Qorinchalar fibrillyatsiyasi / titrashi"),
    "st_elevation": ("STEMI (подозрение)", "STEMI (shubha)"),
    "long_qt": ("Длинный QT — риск TdP", "Uzun QT — TdP xavfi"),
    "peaked_t_waves": ("Паттерн гиперкалиемии (остроконечные T)",
                       "Giperkaliyemiya patterni (uchli T)"),
    "complete_av_block": ("Полная АВ-блокада", "To'liq AV blokada"),
    "ventricular_tachycardia": ("Пробежка широких комплексов (исключить ЖТ)",
                                "Keng komplekslar yugurishi (JT ni istisno qilish)"),
    "wide_complex_tachycardia": ("Тахикардия с широкими комплексами (исключить ЖТ)",
                                 "Keng kompleksli taxikardiya (JT ni istisno qilish)"),
}


SLOW_HINT = ((" — возможна дисфункция синусового узла (СССУ) или АВ-блокада: Холтер, пересмотр препаратов, "
             "замедляющих ритм"),
             (" — sinus tugunining disfunksiyasi (SSSU) yoki AV blokada bo'lishi mumkin: Xolter, ritmni sekinlashtiruvchi "
             "dorilarni qayta ko'rib chiqish"))   # D327
VF_REGULAR_ALERT = ("Правильная тахикардия — исключить ЖТ / ФЖ (D324)",
                    "To'g'ri ritmli taxikardiya — JT / qorinchalar fibrillyatsiyasini istisno qilish")
LONG_QT_CRITICAL_MS = 500.0   # D241: QTc (as the long-QT rule judged it) that raises the critical alert


def _evidence_float(f, feature: str) -> float | None:
    if f is None:
        return None
    for e in f.evidence:
        if e.feature == feature:
            try:
                return float(e.value)
            except (TypeError, ValueError):
                return None
    return None


@dataclass
class CriticalAlert:
    code: str
    label_ru: str
    label_uz: str
    reason: str
    log_id: str = ""
    unverified: bool = False     # raised on a signal that failed the quality gates (D135)


@dataclass
class SafetyResult:
    alerts: list[CriticalAlert] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return bool(self.alerts)


def scan_critical(findings: FindingSet, m: Measurements) -> SafetyResult:
    """Detect critical findings from findings + raw measurements (TZ §10)."""
    result = SafetyResult()

    def add(code: str, ru: str, uz: str, reason: str) -> None:
        alert = CriticalAlert(code=code, label_ru=ru, label_uz=uz, reason=reason,
                              log_id=f"crit-{len(result.alerts) + 1}")
        result.alerts.append(alert)
        result.audit_log.append(f"[{alert.log_id}] CRITICAL {code}: {reason}")
        f = findings.by_code(code)
        if f is not None:
            f.critical = True

    # Finding-derived critical codes.
    for code, (ru, uz) in CRITICAL_CODES.items():
        if code not in findings.codes():
            continue
        if code == "long_qt":
            # D241 (owner, 11.11): the alert from QTc >= 500 ms (AHA/ACCF 2010, ESC 2022: the high-risk
            # threshold for torsades); 450/460-499 stays a non-critical finding in the report.
            qtc = _evidence_float(findings.by_code(code), "qtc_ms")
            if qtc is not None and qtc < LONG_QT_CRITICAL_MS:
                continue
            add(code, ru, uz, reason=f"finding '{code}' present, QTc {qtc} ms >= {LONG_QT_CRITICAL_MS:g}")
            continue
        if code == "ventricular_fibrillation" and any(
                e.feature == "vf_pattern" and e.value == "regular_harmonic" for e in findings.by_code(code).evidence):
            # D324: VF-like waves at a harmonic of a regular R train — the alarm stays, the wording says what is seen
            ru, uz = VF_REGULAR_ALERT
        add(code, ru, uz, reason=f"finding '{code}' present")

    # Measurement-derived critical patterns (may exist without a Layer-4 finding).
    # D327: a slow rate / long pause on one ECG cannot tell sinus node dysfunction from AV block (D326); the alert
    # names both and the next step. Not under pacing: there a pause is a question of the device.
    paced = "paced_rhythm" in findings.codes()
    slow_ru, slow_uz = ("", "") if paced else SLOW_HINT
    hr = m.heart_rate_bpm
    if hr is not None and hr < 40:
        add("severe_bradycardia", "Тяжёлая брадикардия" + slow_ru, "Og'ir bradikardiya" + slow_uz,
            reason=f"HR {hr} bpm < 40")
    # Backstop when Layer 4 did not emit the finding (e.g. pediatric norms): the same
    # 140/min threshold as the adult SVT/WCT split, so nothing wide in 140-150 slips by.
    if (hr is not None and hr >= 140 and (m.intervals.qrs_ms or 0) > 120
            and "wide_complex_tachycardia" not in findings.codes()):
        add("wide_complex_tachycardia",
            "Тахикардия с широкими комплексами (исключить ЖТ)",
            "Keng kompleksli taxikardiya (JT ni istisno qilish)",
            reason=f"HR {hr} bpm with wide QRS {m.intervals.qrs_ms} ms")
    # Significant pause / asystole: longest RR gap >= 3 s (measured in Layer 3).
    if m.max_rr_ms is not None and m.max_rr_ms >= 3000:
        add("ventricular_pause", "Значимая пауза / асистолия" + slow_ru, "Muhim pauza / asistoliya" + slow_uz,
            reason=f"max RR gap {m.max_rr_ms:.0f} ms >= 3000 ms")

    if not result.alerts:
        result.audit_log.append("[safety] no critical findings")
    return result
