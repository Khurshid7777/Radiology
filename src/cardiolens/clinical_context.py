"""Advice from the symptoms the user ticks (D195).

A rule table, not the LLM: it never changes a finding, an alert or a measurement, and it only ever
raises urgency. The symptoms stay out of the narrative's input.
"""

from __future__ import annotations

from dataclasses import dataclass

SYMPTOMS = ("chest_pain", "dyspnea", "syncope", "palpitations")
ONSETS = ("lt12h", "12to48h", "gt48h", "unknown")

SYMPTOM_RU = {"chest_pain": "боль в груди", "dyspnea": "одышка", "syncope": "обморок",
              "palpitations": "сердцебиение"}
SYMPTOM_UZ = {"chest_pain": "ko'krakda og'riq", "dyspnea": "hansirash", "syncope": "hushdan ketish",
              "palpitations": "yurak urishi"}
ONSET_RU = {"lt12h": "менее 12 ч назад", "12to48h": "12–48 ч назад", "gt48h": "более 48 ч назад",
            "unknown": "время начала неизвестно"}
ONSET_UZ = {"lt12h": "12 soatdan kam oldin", "12to48h": "12–48 soat oldin", "gt48h": "48 soatdan ko'p oldin",
            "unknown": "boshlanish vaqti noma'lum"}

# Findings that, with chest pain, mean acute coronary syndrome until proven otherwise.
ISCHEMIA = frozenset({"st_elevation", "st_elevation_any", "st_elevation_nonspecific", "st_depression",
                      "st_depression_minor", "t_inversion", "pathological_q_waves"})
ISCHEMIA_ORDER = ("st_elevation", "st_depression", "st_elevation_any", "pathological_q_waves", "t_inversion",
                  "st_elevation_nonspecific", "st_depression_minor")
# Findings that can explain a faint by an arrhythmia.
SYNCOPE_RHYTHM = frozenset({"long_qt", "short_qt", "complete_av_block", "sinus_bradycardia",
                            "wpw_preexcitation", "brugada_pattern", "wide_complex_tachycardia",
                            "supraventricular_tachycardia", "lbbb", "rbbb"})
SYNCOPE_ORDER = ("complete_av_block", "wide_complex_tachycardia", "long_qt", "brugada_pattern",
                 "wpw_preexcitation", "short_qt", "supraventricular_tachycardia", "sinus_bradycardia", "lbbb",
                 "rbbb")
# An arrhythmia on the record answers palpitations itself.
ARRHYTHMIA = frozenset({"atrial_fibrillation", "atrial_flutter", "supraventricular_tachycardia",
                        "wide_complex_tachycardia", "irregular_rhythm", "ventricular_ectopy",
                        "wpw_preexcitation"})
UNREADABLE = frozenset({"photo_unreliable", "no_ecg_signal", "amplitude_implausible"})
# Symptoms that send mode B to a doctor whatever the record shows.
ESCALATING = frozenset({"chest_pain", "syncope"})


@dataclass
class Advice:
    ru: str
    uz: str
    urgent: bool


def clean(symptoms: list[str] | None, onset: str | None) -> tuple[list[str], str | None]:
    """Known symptom codes only, in a stable order; an unknown onset is dropped."""
    got = set(symptoms or [])
    return [s for s in SYMPTOMS if s in got], (onset if onset in ONSETS else None)


def advise(symptoms: list[str], onset: str | None, codes: set[str], labels: dict[str, str]) -> list[Advice]:
    """``codes``: the findings as shown; ``labels``: code -> Russian label (for naming them)."""
    out: list[Advice] = []
    unreadable = bool(codes & UNREADABLE)

    def named(found: set[str], order: tuple[str, ...]) -> str:
        """The leading finding only: several ST labels repeat each other (D195)."""
        first = next((c for c in order if c in found), min(found))
        return labels.get(first, first)

    if "chest_pain" in symptoms:
        isch = codes & ISCHEMIA
        if unreadable:
            out.append(Advice(
                "Боль в груди, а ЭКГ на снимке не прочитана: не ждите — снимите ЭКГ заново и обратитесь за "
                "медицинской помощью (при сильной или продолжающейся боли — скорая помощь).",
                "Ko'krakda og'riq, suratdagi EKG o'qilmadi: kutmang — EKGni qayta oling va tibbiy yordamga "
                "murojaat qiling (kuchli yoki davom etayotgan og'riqda — tez yordam).", True))
        elif isch:
            early = (" Боль началась менее 12 ч назад — время критично для восстановления кровотока."
                     if onset == "lt12h" else "")
            early_uz = (" Og'riq 12 soatdan kam oldin boshlangan — qon oqimini tiklash uchun vaqt muhim."
                        if onset == "lt12h" else "")
            out.append(Advice(
                f"Боль в груди и изменения ST-T ({named(isch, ISCHEMIA_ORDER)}): действовать как при остром коронарном "
                f"синдроме — срочно к врачу, не ждать повторного анализа.{early}",
                f"Ko'krakda og'riq va ST-T o'zgarishlari: o'tkir koronar sindromdagidek harakat qiling — "
                f"zudlik bilan shifokorga, qayta tahlilni kutmang.{early_uz}", True))
        else:
            out.append(Advice(
                "Боль в груди при ЭКГ без признаков ишемии: нормальная ЭКГ не исключает инфаркт. При "
                "сохраняющейся боли повторите ЭКГ через 15–30 минут и обратитесь к врачу.",
                "Ishemiya belgilarisiz EKGda ko'krakda og'riq: normal EKG infarktni istisno qilmaydi. Og'riq "
                "davom etsa, EKGni 15–30 daqiqadan keyin takrorlang va shifokorga murojaat qiling.", True))

    if "syncope" in symptoms:
        rhythm = codes & SYNCOPE_RHYTHM
        if rhythm and not unreadable:
            out.append(Advice(
                f"Обморок и на ЭКГ: {named(rhythm, SYNCOPE_ORDER)} — возможна аритмическая причина, нужна срочная оценка "
                "врачом.",
                "Hushdan ketish va EKGda aritmiya belgisi: aritmik sabab bo'lishi mumkin — shifokor zudlik "
                "bilan baholashi kerak.", True))
        else:
            out.append(Advice(
                "Обморок: нормальная или непрочитанная ЭКГ не исключает аритмию — нужна оценка врача.",
                "Hushdan ketish: normal yoki o'qilmagan EKG aritmiyani istisno qilmaydi — shifokor baholashi "
                "kerak.", True))

    if "palpitations" in symptoms and not (codes & ARRHYTHMIA) and not unreadable:
        out.append(Advice(
            "Сердцебиение, а на ЭКГ аритмии нет: вне эпизода ЭКГ часто нормальна. Чтобы найти причину, нужна "
            "запись во время сердцебиения (холтер или монитор).",
            "Yurak urishi, EKGda aritmiya yo'q: epizoddan tashqari EKG ko'pincha normal. Sababni topish uchun "
            "yurak urishi paytidagi yozuv kerak (xolter yoki monitor).", False))
    return out


def context_line(symptoms: list[str], onset: str | None, lang: str = "ru") -> str | None:
    if not symptoms:
        return None
    names = SYMPTOM_RU if lang == "ru" else SYMPTOM_UZ
    onsets = ONSET_RU if lang == "ru" else ONSET_UZ
    text = ", ".join(names[s] for s in symptoms)
    return text + (f" ({onsets[onset]})" if onset else "")
