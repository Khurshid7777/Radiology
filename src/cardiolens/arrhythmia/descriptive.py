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


BLOCKED_PAC_EARLY = 0.8      # D286: the P in the pause comes before 0.8 of the median RR after the previous R
BLOCKED_PAC_LEADS = ("II", "V1")


def blocked_pac_in_pause(sig, m) -> dict | None:
    """D286: a P wave early in the diastole of the longest RR (after the T, before the next QRS) in II or V1 — the
    pause follows a non-conducted (blocked) atrial premature beat, not sinus arrest. None when not found."""
    from scipy.signal import butter, filtfilt, find_peaks

    rp = np.asarray(m.r_peaks_samples or [], int)
    if rp.size < 4:
        return None
    fs = sig.sampling_rate_hz
    rr = np.diff(rp)
    i = int(np.argmax(rr))
    med = float(np.median(rr))
    qrs = (m.intervals.qrs_ms or 90.0) / 1000.0
    qt = (m.intervals.qt_ms or 400.0) / 1000.0
    t_end = int((qt - qrs / 2 + 0.02) * fs)
    a, b = int(rp[i] + t_end), int(rp[i + 1] - (qrs / 2 + 0.02) * fs)
    early = int(rp[i] + BLOCKED_PAC_EARLY * med)
    if b - a < int(0.12 * fs) or early <= a:
        return None
    nyq = fs / 2.0
    bb, aa = butter(2, [1.0 / nyq, min(25.0 / nyq, 0.99)], btype="band")
    for ln in BLOCKED_PAC_LEADS:
        if ln not in sig.lead_names:
            continue
        y = filtfilt(bb, aa, np.asarray(sig.lead(ln), float))
        # noise level from the ordinary diastoles (all but the pause)
        dia = [y[int(rp[k] + t_end):int(rp[k + 1] - (qrs / 2 + 0.02) * fs)] for k in range(rr.size) if k != i]
        dia = [d - np.median(d) for d in dia if d.size > int(0.05 * fs)]
        if not dia:
            continue
        d = np.concatenate(dia)
        h = max(2.0 * 1.4826 * float(np.median(np.abs(d - np.median(d)))), 15.0)
        seg = y[a:b] - np.median(y[a:b])
        for pol in (1.0, -1.0):
            pk, _ = find_peaks(pol * seg, height=h, prominence=h, width=(int(0.02 * fs), int(0.16 * fs)))
            hit = [int(a + p) for p in pk if a + p <= early]
            if hit:
                return {"lead": ln, "p_at_ms": round((hit[0] - rp[i]) / fs * 1000.0), "rr_med_ms": round(med / fs * 1000)}
    return None


AVB21_HR_MAX = 60.0
AVB21_CV_MAX = 0.10
AVB21_SHARE_MIN = 0.7      # D292: share of diastoles holding two P waves about half an RR apart
AVB21_HALF_TOL = 0.2


def two_p_share(sig, m, lead: str = "II") -> float | None:
    """D292: share of RR intervals (QRS end .. next QRS onset) with two P-like waves of the same sign about half an
    RR apart — the atria beating twice per conducted beat (2 : 1 AV block). None when it cannot be judged."""
    from scipy.signal import butter, filtfilt, find_peaks

    rp = np.asarray(m.r_peaks_samples or [], int)
    if rp.size < 4 or lead not in sig.lead_names:
        return None
    fs = sig.sampling_rate_hz
    qrs = (m.intervals.qrs_ms or 90.0) / 1000.0
    nyq = fs / 2.0
    bb, aa = butter(2, [1.0 / nyq, min(25.0 / nyq, 0.99)], btype="band")
    y = filtfilt(bb, aa, np.asarray(sig.lead(lead), float))
    qt = (m.intervals.qt_ms or 400.0) / 1000.0
    t_end = qt - qrs / 2 + 0.02          # after the T wave: a T is not a blocked P
    wins = [(int(rp[i] + t_end * fs), int(rp[i + 1] - (qrs / 2 + 0.02) * fs), int(rp[i + 1] - rp[i]))
            for i in range(rp.size - 1)]
    wins = [(a, b, rr) for a, b, rr in wins if b - a > int(0.25 * fs) and a >= 0 and b <= y.size]
    if len(wins) < 3:
        return None
    hits = 0
    for a, b, rr in wins:
        seg = y[a:b] - np.median(y[a:b])
        h = max(2.0 * 1.4826 * float(np.median(np.abs(seg))), 15.0)
        ok = False
        for pol in (1.0, -1.0):
            pk, _ = find_peaks(pol * seg, height=h, prominence=h, width=(int(0.02 * fs), int(0.14 * fs)),
                               distance=int(0.2 * fs))
            if len(pk) >= 2:
                gaps = np.diff(pk) / rr
                if np.any(np.abs(gaps - 0.5) <= AVB21_HALF_TOL):
                    ok = True
        hits += ok
    return hits / len(wins)


AVB21_SHARE_FINAL = 0.9    # D292: chosen on the first half of MIT-BIH 231 "(BII" windows


def avb_2to1_finding(sig, m, codes: set[str]) -> Finding | None:
    """D292: two P waves per RR, about half an RR apart, in a regular bradycardia — probable 2 : 1 AV block (the
    ventricular rate is half the atrial one, so it reads as sinus bradycardia). Descriptive, not an alert."""
    hr, cv = m.heart_rate_bpm, _rr_cv(m.r_peaks_samples)
    if (codes & (NOT_ORGANISED | {"complete_av_block"}) or hr is None or cv is None
            or not (hr < AVB21_HR_MAX and cv < AVB21_CV_MAX)):
        return None
    share = two_p_share(sig, m, "II")
    if share is None or share < AVB21_SHARE_FINAL:
        return None
    return Finding(
        code="probable_av_block_2to1", group="rhythm",
        label_ru=f"Вероятна АВ-блокада 2 : 1: зубцов P вдвое больше, чем QRS (ЧСС желудочков {hr:.0f}) — сверьте с "
                 "лентой; при подтверждении — к кардиологу",
        label_uz=f"AV blokada 2 : 1 ehtimoli: P tishlari QRS dan ikki baravar ko'p (qorinchalar YuQS {hr:.0f}) — "
                 "lenta bilan solishtiring; tasdiqlansa — kardiologga",
        evidence=[Evidence(feature="two_p_share", value=f"{share:.2f}"), Evidence(feature="hr_bpm", value=f"{hr:.0f}")],
        confidence=0.55)
