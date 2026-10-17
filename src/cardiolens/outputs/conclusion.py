"""Bilingual (RU/UZ) textual conclusion (TZ §11.2).

Formats the interpretation report into a human-readable conclusion in Russian
and Uzbek, with the visible reasoning trace (TZ §9 "Reasoning trace обязателен").
"""

from __future__ import annotations

from cardiolens.pipeline import PipelineResult

_MODE_LABEL = {
    "A": ("Режим A — поддержка решений врача", "A rejim — shifokor qarorini qo'llab-quvvatlash"),
    "B": ("Режим B — автономная интерпретация", "B rejim — avtonom talqin"),
    "escalate": ("ТРЕБУЕТСЯ ПРОСМОТР ВРАЧА", "SHIFOKOR KO'RIGI TALAB QILINADI"),
}


# ECG scanner SD-015 / TZ §12: a mark on every report built from a photo.
DIGITIZED_NOTE = (
    "⚠ Получено по фото ЭКГ, возможны искажения сигнала.",
    "⚠ EKG fotosuratidan olingan, signal buzilishi mumkin.",
)


def format_conclusion(result: PipelineResult, lang: str = "ru") -> str:
    r = result.report
    is_ru = lang == "ru"
    lines: list[str] = []
    mode_ru, mode_uz = _MODE_LABEL.get(r.mode, (r.mode, r.mode))
    lines.append(mode_ru if is_ru else mode_uz)
    if result.signal.provenance.digitized:
        lines.append(DIGITIZED_NOTE[0] if is_ru else DIGITIZED_NOTE[1])
    if r.escalation_reason:
        lines.append(("Причина: " if is_ru else "Sabab: ") + r.escalation_reason)
    lines.append("")

    lines.append("НАХОДКИ / TOPILMALAR:" if is_ru else "TOPILMALAR:")
    for c in r.summary:
        lines.append("  • " + (c.text_ru if is_ru else c.text_uz))

    if r.causal_explanation:
        lines.append("")
        lines.append("ПРИЧИННО-СЛЕДСТВЕННЫЙ РАЗБОР:" if is_ru else "SABAB-OQIBAT TAHLILI:")
        for c in r.causal_explanation:
            lines.append("  • " + (c.text_ru if is_ru else c.text_uz))

    if r.red_flags:
        lines.append("")
        lines.append("⚠ КРАСНЫЕ ФЛАГИ / QIZIL BAYROQLAR:")
        for c in r.red_flags:
            lines.append("  ⚠ " + (c.text_ru if is_ru else c.text_uz))

    # Serial comparison against a prior ECG (TZ §8.4).
    if result.serial and result.serial.deltas:
        lines.append("")
        lines.append("ДИНАМИКА vs ПРОШЛАЯ ЭКГ:" if is_ru else "OLDINGI EKGga NISBATAN DINAMIKA:")
        for d in result.serial.deltas:
            mark = "⚠ " if d.critical else "• "
            lines.append("  " + mark + (d.label_ru if is_ru else d.label_uz))
        if result.serial.qtc_delta_ms is not None:
            lines.append(f"  QTcΔ = {result.serial.qtc_delta_ms} мс"
                         if is_ru else f"  QTcΔ = {result.serial.qtc_delta_ms} ms")

    lines.append("")
    conf_word = r.confidence_word_ru if is_ru else r.confidence_word_uz
    lines.append(
        (f"Уверенность: {r.confidence_value:.2f} ({conf_word})") if is_ru
        else (f"Ishonch: {r.confidence_value:.2f} ({conf_word})")
    )
    lines.append(("Следующий шаг: " if is_ru else "Keyingi qadam: ")
                 + (r.next_step_ru if is_ru else r.next_step_uz))

    lines.append("")
    lines.append("РАССУЖДЕНИЕ / MULOHAZA:" if is_ru else "MULOHAZA:")
    for t in r.reasoning_trace:
        lines.append("  - " + t)

    lines.append("")
    lines.append(r.disclaimer_ru if is_ru else r.disclaimer_uz)
    return "\n".join(lines)
