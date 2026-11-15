"""Descriptive wave-5 findings (D282-D284): electrical alternans, short PR without pre-excitation, ectopic (low)
atrial rhythm. None is critical; each names what to check. Thresholds: DECISIONS D282, D283, D284."""
from __future__ import annotations

import numpy as np

from cardiolens.arrhythmia.alternans import detect_alternans
from cardiolens.contracts.findings import Evidence, Finding
from cardiolens.layer3_measure.p_chain import p_signed_cover

SINUS = frozenset({"sinus_rhythm", "sinus_tachycardia", "sinus_bradycardia", "sinus_arrhythmia"})
NOT_ORGANISED = frozenset({"atrial_fibrillation", "possible_atrial_fibrillation", "atrial_flutter", "paced_rhythm",
                           "ventricular_tachycardia", "wide_complex_tachycardia", "wide_qrs_tachycardia",
                           "ventricular_fibrillation", "irregular_rhythm"})
ECTOPY = frozenset({"ventricular_ectopy", "ventricular_bigeminy", "ventricular_trigeminy", "ventricular_couplets",
                    "supraventricular_ectopy", "atrial_bigeminy", "atrial_trigeminy"})
SHORT_PR_MAX_MS = 120.0
SHORT_PR_QRS_MAX_MS = 110.0
SHORT_PR_SD_MAX_MS = 20.0
EAR_NEG_MIN = 0.6
EAR_POS_MAX = 0.1          # D284: 0.2 gave NORM 0.53 % on the PTB-XL rhythm bench, 0.1 gives 0.21 %
EAR_CV_MAX = 0.08
EAR_HR = (50.0, 110.0)
EAR_PR_MIN_MS = 100.0


def _rr_cv(rp) -> float | None:
    rr = np.diff(np.asarray(rp or [], float))
    return float(rr.std() / rr.mean()) if rr.size > 2 and rr.mean() > 0 else None


def alternans_finding(sig, m, codes: set[str]) -> Finding | None:
    if codes & (NOT_ORGANISED | ECTOPY):
        return None
    ok, det = detect_alternans(sig.signal, list(sig.lead_names), m.r_peaks_samples or [], sig.sampling_rate_hz)
    if not ok:
        return None
    leads = ", ".join(det["hits"][:6])
    return Finding(
        code="electrical_alternans", group="rhythm",
        label_ru=f"Электрическая альтернация QRS (амплитуда через удар выше / ниже: {leads}) — при тахикардии "
                 "исключить выпот в перикард (ЭхоКГ)",
        label_uz=f"QRS elektr alternatsiyasi (amplituda har ikkinchi zarbada: {leads}) — taxikardiyada perikard "
                 "suyuqligini istisno qilish (ExoKG)",
        evidence=[Evidence(feature="alternans_leads", value=leads),
                  Evidence(feature="rr_alternation", value=str(det.get("rr_flip")))],
        confidence=0.6)


def short_pr_finding(m, codes: set[str], adult: bool) -> Finding | None:
    pr, qrs, sd = m.intervals.pr_ms, m.intervals.qrs_ms, m.pr_std_ms
    if not adult or pr is None or qrs is None or sd is None:
        return None
    if not (codes & SINUS) or codes & (NOT_ORGANISED | {"wpw_preexcitation"}):
        return None
    if not (pr < SHORT_PR_MAX_MS and qrs < SHORT_PR_QRS_MAX_MS and sd < SHORT_PR_SD_MAX_MS):
        return None
    return Finding(
        code="short_pr", group="interval",
        label_ru=f"Короткий PR ({pr:.0f} мс) без явной дельта-волны — ускоренное АВ-проведение или скрытое предвозбуждение; "
                 "при приступах сердцебиения — к кардиологу",
        label_uz=f"Qisqa PR ({pr:.0f} ms) aniq delta to'lqinsiz — tezlashgan AV o'tkazuvchanlik yoki yashirin oldindan "
                 "qo'zg'alish; yurak urishi xurujlarida — kardiologga",
        evidence=[Evidence(feature="pr_ms", value=f"{pr:.0f}"), Evidence(feature="qrs_ms", value=f"{qrs:.0f}")],
        confidence=0.6)


def ectopic_atrial_finding(sig, m, codes: set[str]) -> Finding | None:
    hr, qrs = m.heart_rate_bpm, m.intervals.qrs_ms
    cv = _rr_cv(m.r_peaks_samples)
    if codes & NOT_ORGANISED or hr is None or qrs is None or cv is None:
        return None
    if not (EAR_HR[0] <= hr <= EAR_HR[1] and qrs < 120 and cv < EAR_CV_MAX):
        return None
    covers = {}
    for ln in ("II", "aVF"):
        if ln not in sig.lead_names:
            return None
        c = p_signed_cover(sig.lead(ln), m.r_peaks_samples or [], sig.sampling_rate_hz, qrs, m.intervals.qt_ms)
        if c is None or not (c["neg"] >= EAR_NEG_MIN and c["pos"] < EAR_POS_MAX
                             and (c["neg_pr"] or 0) >= EAR_PR_MIN_MS):
            return None
        covers[ln] = c
    return Finding(
        code="ectopic_atrial_rhythm", group="rhythm",
        label_ru="Зубец P отрицательный в II и aVF перед каждым QRS — эктопический (нижнепредсердный) ритм; "
                 "проверьте расположение электродов рук",
        label_uz="P tishi II va aVF da har bir QRS oldidan manfiy — ektopik (pastki bo'lmacha) ritm; qo'l "
                 "elektrodlari joylashuvini tekshiring",
        evidence=[Evidence(feature=f"p_neg_share_{ln}", value=str(c["neg"])) for ln, c in covers.items()],
        confidence=0.55)
