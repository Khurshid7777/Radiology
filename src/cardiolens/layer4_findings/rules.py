"""Layer 4 — findings rule engine (TZ §7).

Turns Layer-3 measurements into evidence-backed, bilingual findings across the
MVP subset of groups: rhythm, intervals/QT, ischemia (ST/T), hypertrophy (LVH),
plus morphology/artifact codes that feed the Layer-5 causal engine.

Thresholds come from the age/sex norm tables (TZ §6) — not hard-coded. Findings
this layer cannot support are simply not emitted (TZ §0, no fabrication).
"""

from __future__ import annotations

from typing import Any

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer4_findings.hypertrophy import hypertrophy_findings

# Precordial leads where juvenile T-inversion is a normal variant in children.
_PEDIATRIC_TINV_OK = {"V1", "V2", "V3", "V4"}

# Coronary territories and their reciprocal leads (TZ §7.2, §8.1).
_TERRITORIES = [
    ("нижняя", "pastki", {"II", "III", "aVF"}, {"I", "aVL"}),
    ("передне-перегородочная", "old-septal", {"V1", "V2", "V3", "V4"}, {"II", "III", "aVF"}),
    ("боковая", "yon", {"I", "aVL", "V5", "V6"}, {"III", "aVF"}),
]


def _territory_of(elevated: set[str]) -> tuple[str, str, set[str]] | None:
    """Return (label_ru, label_uz, reciprocal_leads) for the best-matching territory."""
    best = None
    best_overlap = 0
    for ru, uz, leads, recip in _TERRITORIES:
        overlap = len(elevated & leads)
        if overlap >= 2 and overlap > best_overlap:
            best, best_overlap = (ru, uz, recip), overlap
    return best


def _ev(**kv: Any) -> list[Evidence]:
    return [Evidence(feature=k, value=str(v)) for k, v in kv.items()]


def _interval_norms(norms: dict[str, Any], patient) -> dict[str, Any]:
    """The ``intervals_ms`` table, replaced by the patient's age band when the norm
    table has ``intervals_ms_by_age`` (pediatric, D44)."""
    base = dict(norms.get("intervals_ms", {}))
    bands = norms.get("intervals_ms_by_age") or []
    age_days = getattr(patient, "age_days", None)
    age_years = getattr(patient, "age_years", None)
    for band in bands:
        if "age_days_max" in band and age_days is not None and age_days <= band["age_days_max"]:
            base.update({k: v for k, v in band.items() if not k.startswith("age_")})
            return base
        if "age_years_max" in band and age_years is not None and age_years <= band["age_years_max"]:
            base.update({k: v for k, v in band.items() if not k.startswith("age_")})
            return base
    return base


def _axis_range(norms: dict[str, Any], patient) -> tuple[float, float] | None:
    """Normal QRS-axis range, by age band when the table has ``axis_deg_by_age``.

    A rightward axis is physiological in infants and the pediatric table once
    carried one range to 180° for every child; with it, 35 cohort children aged
    4–15 labelled "right axis deviation" (measured 88–134°) scored 0/35 (D53).
    Band = first entry whose age limit the patient is under; no match (or no
    age) falls back to ``axis_deg.qrs_normal``.
    """
    ad = norms.get("axis_deg", {})
    base = ad.get("qrs_normal")
    bands = norms.get("axis_deg_by_age") or []
    age_days = getattr(patient, "age_days", None)
    age_years = getattr(patient, "age_years", None)
    for band in bands:
        if "age_days_max" in band and age_days is not None and age_days <= band["age_days_max"]:
            return tuple(band["qrs_normal"])
        if "age_years_max" in band and age_years is not None and age_years <= band["age_years_max"]:
            return tuple(band["qrs_normal"])
    return tuple(base) if base else None


def _hr_range(norms: dict[str, Any], patient) -> tuple[float, float] | None:
    """Resolve HR norm range, honouring pediatric age-banding (TZ §6)."""
    if "heart_rate_bpm" in norms:
        lo, hi = norms["heart_rate_bpm"]
        return float(lo), float(hi)
    bands = norms.get("heart_rate_bpm_by_age", [])
    age_days = patient.age_days
    age_years = patient.age_years
    for band in bands:
        if "age_days_max" in band and age_days is not None and age_days <= band["age_days_max"]:
            return tuple(band["range"])
        if "age_years_max" in band and age_years is not None and age_years <= band["age_years_max"]:
            return tuple(band["range"])
    if bands:
        return tuple(bands[-1]["range"])
    return None


def detect_findings(
    sig: SignalRepresentation,
    m: Measurements,
    *,
    norms: dict[str, Any],
    la_ra_reversed: bool = False,
) -> FindingSet:
    fs = FindingSet()
    is_pediatric = norms.get("age_group") == "pediatric"

    # ------------------------------------------------------------------ #
    # Rhythm (TZ §7.1)
    # ------------------------------------------------------------------ #
    hr = m.heart_rate_bpm
    hr_range = _hr_range(norms, sig.patient)
    if hr is not None and hr_range is not None:
        lo, hi = hr_range
        if hr < lo:
            fs.findings.append(Finding(
                code="sinus_bradycardia", group="rhythm",
                label_ru=f"Брадикардия (ЧСС {hr}/мин)",
                label_uz=f"Bradikardiya (YUT {hr}/daq)",
                evidence=_ev(heart_rate_bpm=hr, norm_low=lo), confidence=0.9,
            ))
        elif hr > hi:
            # At/above svt_rate_min (adults) a regular narrow tachycardia is SVT, and a
            # wide one is VT until proven otherwise (§10) — not sinus tachycardia; on
            # INCART a 175/min VT run was reported as sinus tachycardia (ported from
            # claude/vibrant-curie-ro1zsz D37). Children have no svt_rate_min.
            svt_min = norms.get("svt_rate_min")
            block = norms.get("intervals_ms", {}).get("qrs_bbb") or 120
            qrs = m.intervals.qrs_ms
            if not is_pediatric and svt_min and hr >= svt_min and qrs is not None and qrs >= block:
                fs.findings.append(Finding(
                    code="wide_complex_tachycardia", group="rhythm",
                    label_ru=f"Тахикардия с широкими QRS (ЧСС {hr}/мин, QRS {qrs} мс) — "
                             f"исключить желудочковую тахикардию",
                    label_uz=f"Keng QRS li taxikardiya (YUT {hr}/daq, QRS {qrs} ms) — "
                             f"ventrikulyar taxikardiyani istisno qilish",
                    evidence=_ev(heart_rate_bpm=hr, qrs_ms=qrs, rate_min=svt_min,
                                 qrs_min=block), confidence=0.7,
                ))
            elif (not is_pediatric and svt_min and hr >= svt_min
                    and m.rhythm_regular is not False and qrs is not None and qrs < block):
                fs.findings.append(Finding(
                    code="supraventricular_tachycardia", group="rhythm",
                    label_ru=f"Суправентрикулярная тахикардия (ЧСС {hr}/мин, "
                             f"регулярная узкокомплексная)",
                    label_uz=f"Supraventrikulyar taxikardiya (YUT {hr}/daq)",
                    evidence=_ev(heart_rate_bpm=hr, svt_rate_min=svt_min,
                                 qrs_ms=qrs, regular="yes"), confidence=0.7,
                ))
            else:
                fs.findings.append(Finding(
                    code="sinus_tachycardia", group="rhythm",
                    label_ru=f"Тахикардия (ЧСС {hr}/мин)",
                    label_uz=f"Taxikardiya (YUT {hr}/daq)",
                    evidence=_ev(heart_rate_bpm=hr, norm_high=hi), confidence=0.9,
                ))
        else:
            fs.findings.append(Finding(
                code="sinus_rhythm", group="rhythm",
                label_ru=f"Синусовый ритм (ЧСС {hr}/мин)",
                label_uz=f"Sinus ritmi (YUT {hr}/daq)",
                evidence=_ev(heart_rate_bpm=hr), confidence=0.85,
            ))

    if m.rhythm_regular is False:
        fs.findings.append(Finding(
            code="irregular_rhythm", group="rhythm",
            label_ru="Нерегулярный ритм (исключить ФП)",
            label_uz="Notekis ritm (AF ni istisno qilish)",
            evidence=_ev(rr_std_ms=m.rr_std_ms), confidence=0.6,
        ))

    # ------------------------------------------------------------------ #
    # Intervals / QT (TZ §7.3)
    # ------------------------------------------------------------------ #
    # Atrial fibrillation: irregularly-irregular rhythm with absent P waves (§7.1).
    # Gates come from the norm table; calibrated on 964 ecg-arrhythmia records
    # (train/test halves) and checked on PTB-XL — see DECISIONS D31.
    rh = norms.get("rhythm", {})
    af_cv, af_p = rh.get("af_rr_cv_min", 0.10), rh.get("af_p_ratio_max", 0.5)
    # Conducted P waves exclude AF regardless of RR irregularity: a refined P onset
    # on most beats with a stable PR (sinus arrhythmia, ectopy) is organised atrial
    # activity. Children's P/R amplitude ratio is small, so the amplitude gate
    # alone read pediatric sinus arrhythmia as AF (15 of 64 PTB-XL children).
    conducted_p = ((m.p_onset_ratio or 0.0) >= 0.5 and m.pr_std_ms is not None
                   and m.pr_std_ms < rh.get("af_pr_std_max_ms", 40))
    # Fibrillatory waves as the second route (D58): the amplitude P gate takes
    # coarse f-waves for organised P on a third of the PTB-XL AF records; an
    # irregular rhythm with a high 4-9 Hz residual and no conducted P is AF too.
    af_fw = rh.get("af_fwave_index_min")
    no_p = m.p_wave_ratio is not None and m.p_wave_ratio < af_p
    fwaves = af_fw is not None and m.fwave_index is not None and m.fwave_index >= float(af_fw)
    # ... unless strong f-waves are what the P search found before each QRS (D81): at a
    # fast or irregular rate its window spans most of the RR and the "PR" is stable only
    # loosely; a tight PR stays conducted. Adults only: in children the conducted-P gate
    # is what keeps sinus arrhythmia from reading as AF.
    # A measured flutter rate means those waves are flutter F waves conducted with a
    # varying ratio, not fibrillation: the override stays off (PTB-XL 1773, D82).
    af_override = rh.get("af_fwave_conducted_override")
    fl_lo, fl_hi = rh.get("atrial_flutter_bpm", [240, 350])
    flutter_rate = m.atrial_rate_bpm is not None and fl_lo <= m.atrial_rate_bpm <= fl_hi
    if (conducted_p and af_override is not None and m.fwave_index is not None
            and m.fwave_index >= float(af_override) and not flutter_rate
            and m.pr_std_ms >= rh.get("af_fwave_override_min_pr_std_ms", 15)):
        conducted_p = False
    if (m.rr_cv or 0) > af_cv and (no_p or fwaves) and not conducted_p:
        fs.findings.append(Finding(
            code="atrial_fibrillation", group="rhythm",
            label_ru="Фибрилляция предсердий (нерегулярный ритм, отсутствие P)",
            label_uz="Bo'lmachalar fibrillyatsiyasi (notekis ritm, P yo'q)",
            evidence=_ev(rr_cv=m.rr_cv, p_wave_ratio=m.p_wave_ratio, pr_std_ms=m.pr_std_ms,
                         fwave_index=m.fwave_index, route="f_waves" if (fwaves and not no_p) else "no_p"),
            confidence=0.7,
        ))

    iv = m.intervals
    _, pr_hi = _interval_norms(norms, sig.patient).get("pr", [None, None])
    # A PR interval needs P waves: organised P (AF gate) or a refined P onset on
    # most beats; and it is meaningless in atrial fibrillation.
    p_present = (m.p_wave_ratio is None or m.p_wave_ratio >= 0.5
                 or (m.p_onset_ratio or 0.0) >= 0.5)
    in_af = "atrial_fibrillation" in fs.codes()
    if (iv.pr_ms is not None and pr_hi and iv.pr_ms > pr_hi and p_present
            and not in_af and m.av_dissociation is not True):
        fs.findings.append(Finding(
            code="first_degree_av_block", group="interval",
            label_ru=f"АВ-блокада I степени (PR {iv.pr_ms} мс)",
            label_uz=f"I darajali AV blokada (PR {iv.pr_ms} ms)",
            evidence=_ev(pr_ms=iv.pr_ms, norm_high=pr_hi), confidence=0.85,
        ))

    _conduction_findings(m, norms, fs, sig.patient)
    _qt_findings(iv, norms, is_pediatric, sig, fs)

    # Complete (third-degree) AV block: atrioventricular dissociation (§7.1, §10).
    if m.av_dissociation is True:
        fs.findings.append(Finding(
            code="complete_av_block", group="rhythm",
            label_ru="Полная АВ-блокада (АВ-диссоциация)",
            label_uz="To'liq AV blokada (AV dissotsiatsiya)",
            evidence=_ev(av_dissociation=True, atrial_rate_bpm=m.atrial_rate_bpm,
                         ventricular_rate_bpm=hr), confidence=0.8,
        ))

    # Atrial flutter: regular atrial rate ~250-350/min (sawtooth) (§7.1).
    fl_lo, fl_hi = norms.get("rhythm", {}).get("atrial_flutter_bpm", [240, 350])
    if (m.atrial_rate_bpm is not None and fl_lo <= m.atrial_rate_bpm <= fl_hi
            and not _hr_harmonic(m, norms.get("rhythm", {}))):
        fs.findings.append(Finding(
            code="atrial_flutter", group="rhythm",
            label_ru=f"Трепетание предсердий (частота предсердий {m.atrial_rate_bpm}/мин)",
            label_uz=f"Bo'lmachalar titrashi (bo'lmacha chastotasi {m.atrial_rate_bpm}/daq)",
            evidence=_ev(atrial_rate_bpm=m.atrial_rate_bpm), confidence=0.65,
        ))

    # A detected non-sinus rhythm (AF / flutter) replaces the sinus_* rate codes:
    # "sinus rhythm at 80" is wrong in AF, and the rate qualifier belongs to the
    # arrhythmia label instead (rapid/slow ventricular response).
    _replace_sinus_codes_if_non_sinus(fs, hr, hr_range)

    # ------------------------------------------------------------------ #
    # Hypertrophy — LVH (TZ §7.4)
    # ------------------------------------------------------------------ #
    # Axis first (Romhilt-Estes and the pediatric criteria use it), then
    # Sokolow-Lyon / Cornell / Romhilt-Estes for adults, age-band voltages for
    # children (layer4_findings/hypertrophy.py).
    _axis_findings(m, norms, fs, sig.patient)
    hypertrophy_findings(m, norms, fs, sig.patient)

    # ------------------------------------------------------------------ #
    # Ischemia — ST / T (TZ §7.2). Runs after conduction + LVH so secondary
    # (discordant) ST-T changes can be told from primary ischemia.
    # ------------------------------------------------------------------ #
    _st_findings(m, norms, fs, sig.patient)
    _t_inversion_findings(m, is_pediatric, fs)

    # Peaked T-waves feed the hyperkalemia causal chain.
    _peaked_t_findings(m, norms, fs)
    if len(m.morphology.flat_t_leads) >= 3:
        fs.findings.append(Finding(
            code="flat_t_waves", group="other",
            label_ru="Уплощение зубцов T",
            label_uz="T to'lqinlarining yassilanishi",
            evidence=_ev(leads=",".join(m.morphology.flat_t_leads)), confidence=0.6,
        ))
    if m.morphology.u_wave_leads:
        fs.findings.append(Finding(
            code="u_waves", group="other",
            label_ru="U-волны",
            label_uz="U to'lqinlar",
            evidence=_ev(leads=",".join(m.morphology.u_wave_leads)), confidence=0.6,
        ))

    # ------------------------------------------------------------------ #
    # Axis deviation (TZ §7.4) and low QRS voltage (TZ §7.5)
    # ------------------------------------------------------------------ #
    _low_voltage_findings(m, norms, fs)
    _atrial_enlargement_findings(m, norms, fs)

    # ------------------------------------------------------------------ #
    # Artifact hypothesis from Layer 2 (TZ §5, feeds §8.3)
    # ------------------------------------------------------------------ #
    # In children a rightward axis is physiological, and lead I then looks
    # "inverted" with an upright aVR: the reversal hypothesis is only raised when
    # the axis is outside the pediatric normal range (D44).
    axis_ok_for_age = True
    if is_pediatric and m.axes.qrs_deg is not None:
        lo_ax, hi_ax = _axis_range(norms, sig.patient) or (-30, 180)
        axis_ok_for_age = lo_ax <= m.axes.qrs_deg <= hi_ax
    if la_ra_reversed and not (is_pediatric and axis_ok_for_age):
        fs.findings.append(Finding(
            code="lead_reversal_la_ra", group="other",
            label_ru="Подозрение на реверс электродов LA/RA",
            label_uz="LA/RA elektrodlar almashinuvi shubhasi",
            evidence=_ev(lead_I="inverted", aVR="upright"), confidence=0.65,
        ))

    # Pathological Q (old MI) and the Brugada type-1 pattern on the median beat (ported
    # from claude/vibrant-curie-ro1zsz D36, D89, D90). LBBB suppresses Brugada as in the source.
    _q_wave_findings(m, norms, fs)
    if fs.by_code("lbbb") is None:
        _brugada_pattern(m, fs)

    # In a wide-complex tachycardia the "P" before each QRS and the initial QRS forces
    # are not interpretable: a short "PR" is not pre-excitation and PR is not AV
    # conduction (ported from claude/vibrant-curie-ro1zsz D37).
    if fs.by_code("wide_complex_tachycardia") is not None:
        fs.findings = [f for f in fs.findings if f.code not in _WCT_EXCLUDES]

    return fs


_WCT_EXCLUDES = ("wpw_preexcitation", "first_degree_av_block", "pathological_q_waves")


def _hr_harmonic(m, rh: dict) -> bool:
    """The "atrial rate" is an HR harmonic of a regular sinus tachycardia (D83, D84): the
    QRS-T train at 2 x HR in the slow-flutter range or with sinus P waves and a quiet
    baseline, or at 3 x HR, with a regular RR; or at 2.5 / 3.5 x HR with a regular RR."""
    top = rh.get("flutter_harmonic_max_bpm")
    hr, rate, mad = m.heart_rate_bpm, m.atrial_rate_bpm, m.rr_mad_cv
    if top is None or hr is None or rate is None or mad is None:
        return False
    tol = rh.get("flutter_harmonic_tolerance", 0.08)
    # Half-integer multiples (2.5, 3.5 x HR) of a regular rhythm: a two-beat cycle's
    # harmonics; flutter there needs alternating conduction, hence alternating RR (D85).
    half_cv = rh.get("flutter_harmonic_half_rr_cv_max")
    if (half_cv is not None and m.rr_cv is not None and m.rr_cv < half_cv
            and any(abs(rate - k * hr) <= tol * k * hr for k in (2.5, 3.5))):
        return True
    if mad >= rh.get("flutter_harmonic_rr_mad_cv_max", 0.03):
        return False
    at2 = abs(rate - 2 * hr) <= tol * 2 * hr
    sinus_p = (m.p_wave_ratio is not None and m.fwave_index is not None
               and m.p_wave_ratio >= rh.get("flutter_harmonic_p_ratio_min", 0.85)
               and m.fwave_index < rh.get("flutter_harmonic_fwave_max", 2.5))
    return ((at2 and (rate < float(top) or sinus_p))
            or abs(rate - 3 * hr) <= tol * 3 * hr)


_SINUS_CODES = {"sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia"}
_NON_SINUS_CODES = {"atrial_fibrillation", "atrial_flutter", "complete_av_block"}


def _replace_sinus_codes_if_non_sinus(fs: FindingSet, hr, hr_range) -> None:
    codes = fs.codes()
    if not (codes & _NON_SINUS_CODES):
        return
    fs.findings = [f for f in fs.findings if f.code not in _SINUS_CODES]
    if hr is None or hr_range is None:
        return
    lo, hi = hr_range
    qual_ru = " с быстрым желудочковым ответом" if hr > hi else (
        " с медленным желудочковым ответом" if hr < lo else "")
    qual_uz = " tez qorincha javobi bilan" if hr > hi else (
        " sekin qorincha javobi bilan" if hr < lo else "")
    for f in fs.findings:
        if f.code in _NON_SINUS_CODES and qual_ru:
            f.label_ru += f"{qual_ru} (ЧСС {hr}/мин)"
            f.label_uz += f"{qual_uz} (YUT {hr}/daq)"


def _qt_findings(iv, norms, is_pediatric, sig, fs: FindingSet) -> None:
    qtc_norms = norms.get("qtc_ms", {})
    # Correction formula from the norm table. Fridericia is the default: Bazett
    # over-corrects at fast rates (real-data long-QT specificity 0.68 vs 0.85).
    formula = str(qtc_norms.get("formula", "fridericia")).lower()
    by_formula = {"bazett": iv.qtc_bazett_ms, "fridericia": iv.qtc_fridericia_ms,
                  "framingham": iv.qtc_framingham_ms}
    qtc = by_formula.get(formula)
    if qtc is None:                      # hand-built Measurements carry only Bazett
        qtc, formula = iv.qtc_bazett_ms, "bazett"
    if qtc is None:
        return
    # Under a wide QRS the QT carries the conduction delay, not repolarisation:
    # two thirds of the long-QT false calls on the 664 set had QRS >= 120 ms
    # (LBBB 36, RBBB 20 of 91). Bogossian's correction (QT - 0.485 x QRS) is
    # applied to the QTc before the threshold; the measured QTc stays as is.
    wide_ms = qtc_norms.get("wide_qrs_ms")
    frac = qtc_norms.get("wide_qrs_qt_fraction")
    adjusted_from = None
    if (wide_ms and frac and iv.qt_ms and iv.qrs_ms
            and (iv.qrs_ms >= float(wide_ms) or fs.codes() & {"lbbb", "rbbb"})):
        qt_adj = iv.qt_ms - float(frac) * iv.qrs_ms
        if qt_adj > 0:
            adjusted_from = qtc
            # Bazett/Fridericia scale linearly with QT; Framingham is additive.
            qtc = round(qtc - (iv.qt_ms - qt_adj) if formula == "framingham" else qtc * qt_adj / iv.qt_ms, 1)
    if is_pediatric:
        upper = qtc_norms.get("upper")
        short_lower = qtc_norms.get("short_lower")
    else:
        sex = (sig.patient.sex or "male").lower()
        upper = qtc_norms.get("female_upper" if sex == "female" else "male_upper")
        short_lower = qtc_norms.get("short_lower")
    if upper and qtc > upper:
        ev = _ev(qtc_ms=qtc, formula=formula, upper=upper)
        if adjusted_from is not None:
            ev += _ev(qtc_measured_ms=adjusted_from, qrs_adjustment="bogossian")
        fs.findings.append(Finding(
            code="long_qt", group="interval",
            label_ru=f"Удлинение QT (QTc {qtc} мс, {formula}"
                     + (", с поправкой на широкий QRS)" if adjusted_from is not None else ")"),
            label_uz=f"QT uzayishi (QTc {qtc} ms, {formula}"
                     + (", keng QRS uchun tuzatilgan)" if adjusted_from is not None else ")"),
            evidence=ev, confidence=0.85,
        ))
    elif short_lower and qtc < short_lower:
        fs.findings.append(Finding(
            code="short_qt", group="interval",
            label_ru=f"Укорочение QT (QTc {qtc} мс, {formula})",
            label_uz=f"QT qisqarishi (QTc {qtc} ms, {formula})",
            evidence=_ev(qtc_ms=qtc, formula=formula, lower=short_lower), confidence=0.7,
        ))


def _territorial_st_elevation(elevated, depressed, elev, fs: FindingSet, proportional=()) -> None:
    """Focal (single-territory) ST elevation → STEMI with reciprocal check (§8.1)."""
    terr = _territory_of(set(elevated))
    if terr:
        ru, uz, recip = terr
        recip_present = sorted(recip & set(depressed))
        recip_txt = f"; реципрокная депрессия в {', '.join(recip_present)}" if recip_present else ""
        ev = _ev(leads=",".join(elevated), territory=ru, reciprocal=",".join(recip_present) or "none")
        if proportional:
            ev += _ev(discordant_proportional=",".join(proportional))
        fs.findings.append(Finding(
            code="st_elevation", group="ischemia",
            label_ru=f"Элевация ST, {ru} территория ({', '.join(elevated)}){recip_txt}",
            label_uz=f"ST elevatsiyasi, {uz} hudud ({', '.join(elevated)})",
            evidence=ev, confidence=0.8,
        ))
    else:
        # Elevation without two contiguous leads of one territory is not a STEMI
        # pattern (early repolarisation, baseline, single-lead artefact): reported
        # for the physician, never critical.
        fs.findings.append(Finding(
            code="st_elevation_nonspecific", group="ischemia",
            label_ru=f"Элевация ST без территориальной картины ({', '.join(elevated)})",
            label_uz=f"Hududiy manzarasiz ST elevatsiyasi ({', '.join(elevated)})",
            evidence=_ev(leads=",".join(elevated), threshold_uv=elev), confidence=0.5,
        ))


_PRECORDIAL = {"V1", "V2", "V3", "V4", "V5", "V6"}


def _peaked_t_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Tall, narrow, symmetric ("tented") T waves in >= 2 leads — the hyperkalaemia
    pattern (TZ §7, §8; D61). Thresholds from the norm table's ``t_wave`` block:
    amplitude (precordial / limb), T/QRS ratio, half-width window and up/down
    asymmetry window. Amplitude alone is not a criterion: in the 664 set the 90th
    percentile of the tallest precordial T is 1.45 mV, and the old amplitude rule
    fired on 21 % of records, half of them on QRS remnants in the T window."""
    tw = norms.get("t_wave", {})
    amp_p, amp_l = tw.get("peaked_min_uv_precordial"), tw.get("peaked_min_uv_limb")
    if not (amp_p and amp_l):
        return
    ratio_min = float(tw.get("peaked_t_qrs_ratio", 0.0))
    hw_lo, hw_hi = tw.get("peaked_half_width_ms", [0, 10**6])
    as_lo, as_hi = tw.get("peaked_asymmetry", [0.0, 10**6])
    leads = []
    for t in m.morphology.t_waves:
        if t.lead == "aVR" or t.half_width_ms is None or t.asymmetry is None:
            continue
        amp_min = float(amp_p if t.lead in _PRECORDIAL else amp_l)
        if (t.amplitude_uv >= amp_min and (t.qrs_ratio or 0.0) >= ratio_min
                and float(hw_lo) <= t.half_width_ms <= float(hw_hi)
                and float(as_lo) <= t.asymmetry <= float(as_hi)):
            leads.append(t.lead)
    m.morphology.peaked_t_leads = leads
    if len(leads) >= int(tw.get("peaked_min_leads", 2)):
        fs.findings.append(Finding(
            code="peaked_t_waves", group="other",
            label_ru="Высокие остроконечные зубцы T",
            label_uz="Baland uchli T to'lqinlar",
            evidence=_ev(leads=",".join(leads),
                         amplitude_uv=",".join(str(round(t.amplitude_uv)) for t in m.morphology.t_waves if t.lead in leads),
                         half_width_ms=",".join(str(round(t.half_width_ms)) for t in m.morphology.t_waves if t.lead in leads)),
            confidence=0.7,
        ))


def _atrial_enlargement_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Left/right atrial enlargement from P-wave duration/amplitude (TZ §7.4)."""
    if fs.codes() & _NON_SINUS_CODES:
        return  # no true P waves to measure in AF / flutter
    pw = norms.get("p_wave", {})
    dur_hi, amp_hi = pw.get("duration_upper_ms"), pw.get("amplitude_upper_uv")
    if dur_hi and m.p_wave_duration_ms is not None and m.p_wave_duration_ms > dur_hi:
        fs.findings.append(Finding(
            code="left_atrial_enlargement", group="hypertrophy",
            label_ru=f"Увеличение левого предсердия (P-mitrale, {m.p_wave_duration_ms} мс)",
            label_uz=f"Chap bo'lmacha kattalashuvi (P-mitrale, {m.p_wave_duration_ms} ms)",
            evidence=_ev(p_wave_duration_ms=m.p_wave_duration_ms, threshold=dur_hi),
            confidence=0.65,
        ))
    if amp_hi and m.p_wave_amplitude_uv is not None and m.p_wave_amplitude_uv > amp_hi:
        fs.findings.append(Finding(
            code="right_atrial_enlargement", group="hypertrophy",
            label_ru=f"Увеличение правого предсердия (P-pulmonale, {m.p_wave_amplitude_uv} мкВ)",
            label_uz=f"O'ng bo'lmacha kattalashuvi (P-pulmonale, {m.p_wave_amplitude_uv} µV)",
            evidence=_ev(p_wave_amplitude_uv=m.p_wave_amplitude_uv, threshold=amp_hi),
            confidence=0.65,
        ))


def _axis_findings(m: Measurements, norms, fs: FindingSet, patient=None) -> None:
    """QRS-axis deviation (TZ §7.4); the normal range is age-banded in children (D53)."""
    axis = m.axes.qrs_deg
    ad = norms.get("axis_deg", {})
    rng = _axis_range(norms, patient)
    if axis is None or not rng:
        return
    lo, hi = rng
    ext_lo, ext_hi = ad.get("extreme_low"), ad.get("extreme_high")
    if ext_lo is not None and (axis < ext_lo or (ext_hi is not None and axis > ext_hi)):
        fs.findings.append(Finding(
            code="extreme_axis", group="hypertrophy",
            label_ru=f"Резкое отклонение оси (ось QRS {axis}°)",
            label_uz=f"O'qning keskin og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis), confidence=0.7,
        ))
    elif axis < lo:
        fs.findings.append(Finding(
            code="left_axis_deviation", group="hypertrophy",
            label_ru=f"Отклонение оси влево (ось QRS {axis}°)",
            label_uz=f"O'qning chapga og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis, norm_low=lo), confidence=0.7,
        ))
    elif axis > hi:
        fs.findings.append(Finding(
            code="right_axis_deviation", group="hypertrophy",
            label_ru=f"Отклонение оси вправо (ось QRS {axis}°)",
            label_uz=f"O'qning o'ngga og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis, norm_high=hi), confidence=0.7,
        ))


def _low_voltage_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Low QRS voltage in the limb leads (TZ §7.5)."""
    thr = norms.get("voltage_uv", {}).get("low_voltage_limb")
    amp = m.qrs_amplitude_limb_uv
    if thr and amp is not None and amp < thr:
        fs.findings.append(Finding(
            code="low_qrs_voltage", group="other",
            label_ru=f"Низкий вольтаж QRS (амплитуда {amp} мкВ)",
            label_uz=f"Past QRS voltaji (amplituda {amp} µV)",
            evidence=_ev(qrs_amplitude_limb_uv=amp, threshold=thr), confidence=0.7,
        ))


def _conduction_findings(m: Measurements, norms, fs: FindingSet, patient=None) -> None:
    """Classify a wide QRS into WPW / RBBB / LBBB / nonspecific (TZ §7.1)."""
    iv = m.intervals
    ivn = _interval_norms(norms, patient)
    qrs_hi = ivn.get("qrs_bbb") or ivn.get("qrs", [None, None])[1]
    pr_lo = ivn.get("pr", [None, None])[0]
    if iv.qrs_ms is None:
        return
    wide = bool(qrs_hi) and iv.qrs_ms > qrs_hi
    # Pre-excitation (WPW), two routes (TZ §8.1): the classic short PR + wide QRS,
    # or a delta wave — a slurred initial QRS (slow slope for >= wpw.delta_ms in
    # the median lead) with a PR not above wpw.pr_max and a QRS not narrower than
    # wpw.qrs_min (D51: PTB-XL WPW 0.35 -> 0.51 at spec 0.981; half of the WPW
    # records have PR 120-140 and QRS 100-115, outside the classic conjunction).
    # An RSR' pattern in V1 with a wide QRS is RBBB, not a delta wave.
    wpw = norms.get("wpw", {})
    delta = m.morphology.delta_ms
    # The classic conjunction alone had a 50 % positive predictive value on
    # PTB-XL (28 true / 28 false) and fired on 52 bundle branch blocks with a
    # PR of 100-118 ms in MIMIC-IV-ECG: a short PR with a wide QRS is also a
    # block in a patient whose PR is short. Pre-excitation slurs the upstroke,
    # so the classic route needs the delta descriptor too, at a lower bar than
    # the delta route (D65: true classic calls had delta 53-90 ms, false 21-55).
    classic_delta = wpw.get("classic_delta_ms")
    classic = (wide and bool(pr_lo) and iv.pr_ms is not None and iv.pr_ms < pr_lo
               and (classic_delta is None or delta is None or delta >= float(classic_delta)))
    # A definite-LBBB width with a PR that is not short is a block whose slow
    # initial activation looks like a delta wave (D65: 13 such false calls in
    # MIMIC at QRS 136-200 ms; the trade on PTB-XL is 4 true calls).
    lbbb_width = ivn.get("qrs_lbbb_definite")
    block_like = (bool(lbbb_width) and bool(pr_lo) and iv.pr_ms is not None
                  and iv.pr_ms >= pr_lo and iv.qrs_ms >= float(lbbb_width))
    by_delta = (delta is not None and wpw.get("delta_ms") is not None and delta >= wpw["delta_ms"]
                and iv.pr_ms is not None and iv.pr_ms <= wpw.get("pr_max", 140)
                and iv.qrs_ms >= wpw.get("qrs_min", 90)
                and not (wide and m.qrs_v1_terminal_positive is True)
                and not block_like)
    if classic or by_delta:
        how = "короткий PR" if classic else f"дельта-волна {delta} мс"
        how_uz = "qisqa PR" if classic else f"delta to'lqin {delta} ms"
        fs.findings.append(Finding(
            code="wpw_preexcitation", group="interval",
            label_ru=f"Преэкзитация (WPW): {how}, PR {iv.pr_ms} мс, QRS {iv.qrs_ms} мс",
            label_uz=f"Preeksitatsiya (WPW): {how_uz}, PR {iv.pr_ms} ms, QRS {iv.qrs_ms} ms",
            evidence=_ev(pr_ms=iv.pr_ms, qrs_ms=iv.qrs_ms, delta_ms=delta, route="classic" if classic else "delta"),
            confidence=0.7 if classic else 0.6,
        ))
        return
    if not wide:
        return
    # BBB typing from the terminal QRS forces in V1 (R' -> RBBB, QS/rS -> LBBB).
    # V6 is only a fallback when V1 is missing: on real LBBB the discordant ST
    # segment leaks into the terminal window and flips the V6 sign.
    v1, v6 = m.qrs_v1_terminal_positive, m.qrs_v6_terminal_positive
    term = v1 if v1 is not None else (None if v6 is None else (not v6))
    # LBBB needs more than "wide + V1 negative": a normal rS in V1 is terminal
    # negative too, and an acute STEMI whose ST take-off merges with the QRS end
    # reads 118-128 ms by the envelope (12SL: 98-112) — that call then silenced
    # the elevation as a block mimic on 37 of 118 missed 12SL-STEMI records (D50).
    # True CLBBB on PTB-XL is 143-179 ms wide (p10-p90) with an intrinsicoid
    # deflection in V5/V6 of 54-102 ms; so: definitely wide, or wide with a late
    # intrinsicoid deflection. Otherwise it is a non-specific wide QRS.
    lbbb_wide = ivn.get("qrs_lbbb_definite")
    lbbb_idt = ivn.get("lbbb_intrinsicoid")
    idt = m.voltages.intrinsicoid_v5_v6_ms if m.voltages is not None else None
    lbbb_support = (lbbb_wide is None or iv.qrs_ms >= lbbb_wide
                    or (lbbb_idt is not None and idt is not None and idt >= lbbb_idt))
    if term is False and not lbbb_support:
        term = None
    if term is True:
        fs.findings.append(Finding(
            code="rbbb", group="interval",
            label_ru=f"Полная блокада правой ножки (QRS {iv.qrs_ms} мс, RSR' в V1)",
            label_uz=f"O'ng oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms, V1 da RSR')",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="positive", v6_terminal=str(v6)),
            confidence=0.72,
        ))
    elif term is False:
        fs.findings.append(Finding(
            code="lbbb", group="interval",
            label_ru=f"Полная блокада левой ножки (QRS {iv.qrs_ms} мс)",
            label_uz=f"Chap oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="negative", v6_terminal="positive"),
            confidence=0.72,
        ))
    else:
        fs.findings.append(Finding(
            code="wide_qrs_nonspecific", group="interval",
            label_ru=f"Расширение QRS ({iv.qrs_ms} мс)",
            label_uz=f"QRS kengayishi ({iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, norm_high=qrs_hi), confidence=0.8,
        ))


_SECONDARY_ST_CONTEXT = {"rbbb", "lbbb", "wpw_preexcitation", "lvh"}
# Contexts whose expected repolarisation can mimic ST ELEVATION (discordant to a
# negative QRS): LBBB, pre-excitation, LVH strain. RBBB is not one of them — its
# secondary change is ST depression / T inversion in V1-V3, so ST elevation with
# an RBBB is primary (acute anterior MI with RBBB was vetoed on PTBDB, D38).
_ELEVATION_MIMIC_CONTEXT = {"lbbb", "wpw_preexcitation", "lvh"}


def _st_thresholds(norms, patient, lead: str) -> tuple[float | None, float | None]:
    """(elevation, depression) thresholds for one lead from the norm table."""
    thr = norms.get("st_uv", {})
    elev, depr = thr.get("elevation_significant"), thr.get("depression_significant")
    v23 = thr.get("elevation_v2_v3")
    if lead == "V4" and thr.get("elevation_v4") and elev:
        elev = float(thr["elevation_v4"])          # pediatric table: V4 has its own 98th percentile (D66)
    if lead in ("V2", "V3") and isinstance(v23, dict) and elev:
        sex = (getattr(patient, "sex", None) or "male").lower()
        age = getattr(patient, "age_years", None)
        if sex == "female":
            elev = v23.get("female", elev)
        elif age is not None and age < 40:
            elev = v23.get("male_under_40", v23.get("male", elev))
        else:
            elev = v23.get("male", elev)
    return elev, depr


def _st_findings(m: Measurements, norms, fs: FindingSet, patient=None) -> None:
    thr = norms.get("st_uv", {})
    if not (thr.get("elevation_significant") and thr.get("depression_significant")):
        return
    plateau = float(thr.get("elevation_plateau_fraction", 0.8))
    upsloping = float(thr.get("depression_upsloping_uv", 50))
    # Descriptor thresholds: a flat 1 mm in any lead (the STEMI-grade sex/age V2-V3
    # thresholds belong to the alert only); discordant elevation under BBB/LVH/WPW
    # counts only when it is large (Sgarbossa-like), see D37.
    any_thr = float(thr.get("any_elevation_uv", thr["elevation_significant"]))
    any_disc_thr = float(thr.get("any_elevation_discordant_uv", 2 * any_thr))
    # Proportional rescue of discordant elevation (Smith-modified Sgarbossa /
    # Armstrong, D54): J >= prop_ratio x the lead's own S depth and >= prop_min_uv.
    prop_ratio = thr.get("proportional_discordant_ratio")
    prop_min = float(thr.get("proportional_discordant_min_uv", thr["elevation_significant"]))
    proportional: list[str] = []
    # With a bundle branch block, pre-excitation or LVH the expected repolarisation
    # is discordant to the QRS; only concordant deviations count as primary.
    secondary_ctx = bool(fs.codes() & _SECONDARY_ST_CONTEXT)
    elevation_mimic_ctx = bool(fs.codes() & _ELEVATION_MIMIC_CONTEXT)
    # Early repolarisation (D66, pediatric table): elevation that is small next to
    # the lead's own T wave and upsloping is the juvenile normal variant.
    er_ratio = thr.get("early_repolarisation_j_t_ratio_max")
    t_amp = {t.lead: t.amplitude_uv for t in m.morphology.t_waves}
    elevated, depressed, discordant, any_elevated, early_repol = [], [], [], [], []
    for st in m.morphology.st:
        elev, depr = _st_thresholds(norms, patient, st.lead)
        j, j60, j80 = st.j_point_uv, st.j60_uv, st.j80_uv
        # Descriptor needs a persisting elevation (J and J+60): a J-point-only bump
        # (early repolarisation, S-wave tail) fired on half of all records.
        disc_elev = elevation_mimic_ctx and st.qrs_net_uv is not None and st.qrs_net_uv < 0
        if (st.lead != "aVR" and j is not None and j >= any_thr
                and (j60 is None or j60 >= plateau * any_thr)
                and (not disc_elev or j >= any_disc_thr)):
            any_elevated.append(st.lead)
        dev = 0
        # STEMI is defined by J-POINT elevation that persists into the ST segment.
        if j is not None and j >= elev and (j60 is None or j60 >= plateau * elev):
            dev = 1
            ta = t_amp.get(st.lead)
            if (er_ratio is not None and ta is not None and ta > 0 and j / ta <= float(er_ratio)
                    and j60 is not None and j60 >= j):
                early_repol.append(st.lead)
                dev = 0
        # Depression is judged in the ST segment (J+60 / J+80), horizontal or
        # downsloping; an upsloping segment (J+80 rising over J+60) is not called.
        elif j60 is not None:
            low = min(j60, j80) if j80 is not None else j60
            if low <= -depr and not (j80 is not None and j80 - j60 >= upsloping):
                dev = -1
                # Pediatric table (D68): a segment descending into this lead's own
                # inverted T wave is the juvenile T pattern's limb, not depression.
                into_t = thr.get("depression_into_inverted_t_margin_uv")
                ta = t_amp.get(st.lead)
                if (into_t is not None and ta is not None and ta < -100.0
                        and j80 is not None and j80 < j60 - float(into_t)):
                    dev = 0
        elif j is not None and j <= -depr:
            dev = -1
        if dev == 0:
            continue
        ctx = elevation_mimic_ctx if dev > 0 else secondary_ctx
        if ctx and st.qrs_net_uv is not None and (st.qrs_net_uv > 0) != (dev > 0):
            # Discordant ELEVATION is the expected repolarisation of a BBB/LVH and
            # must not raise a STEMI alert — unless it is out of proportion to the
            # S wave it follows (>= prop_ratio of the S depth): that is primary
            # elevation on top of the block / hypertrophy (D54). Discordant
            # DEPRESSION is still ST depression (strain pattern) — reported, and
            # explained by the LVH-strain / BBB causal chains.
            if dev > 0:
                s_depth = st.s_depth_uv
                if (prop_ratio is not None and s_depth and s_depth > 0 and j is not None
                        and j >= prop_min and j / s_depth >= float(prop_ratio)):
                    proportional.append(st.lead)
                else:
                    discordant.append(st.lead)
                    continue
            else:
                discordant.append(st.lead)
        (elevated if dev > 0 else depressed).append(st.lead)

    if len(elevated) >= 2:
        # Diffuse ST elevation across >=2 coronary territories (and no single
        # reciprocal territory) points to pericarditis, not a focal STEMI — a
        # classic mimic distinction (ecglibrary / 150 ECG Problems).
        # Proportional discordant leads (D54) rescue a focal alert under a block;
        # they are not evidence of diffuse elevation, so the territory count for
        # the pericarditis pattern uses the primary (concordant) leads only.
        primary = set(elevated) - set(proportional)
        n_terr = sum(1 for _, _, leads, _ in _TERRITORIES if len(primary & leads) >= 2)
        if n_terr >= 2 and len(depressed) == 0:
            fs.findings.append(Finding(
                code="pericarditis", group="other",
                label_ru=f"Диффузная элевация ST — перикардит? ({', '.join(elevated)})",
                label_uz=f"Diffuz ST elevatsiyasi — perikardit? ({', '.join(elevated)})",
                evidence=_ev(leads=",".join(elevated), territories=n_terr,
                             reciprocal="none"), confidence=0.6,
            ))
        else:
            _territorial_st_elevation(elevated, depressed, thr.get("elevation_significant"), fs,
                                      proportional=sorted(set(proportional) & set(elevated)))
    # Discordant depression under a block / LVH / pre-excitation is the expected
    # repolarisation of that QRS and goes to `secondary_st_t_change` (below), not
    # to the ischaemia code: with it inside, `st_depression` had 277 false calls on
    # MIMIC (188 under a BBB) and 344 on PTB-XL (151 BBB, 43 LVH, 39 WPW) — D67.
    primary_depressed = [lead for lead in depressed if lead not in discordant]
    if len(primary_depressed) >= 2:
        disc = sorted(set(depressed) & set(discordant))
        fs.findings.append(Finding(
            code="st_depression", group="ischemia",
            label_ru=f"Депрессия ST ({', '.join(primary_depressed)})",
            label_uz=f"ST depressiyasi ({', '.join(primary_depressed)})",
            evidence=_ev(leads=",".join(primary_depressed), threshold_uv=thr.get("depression_significant"),
                         discordant_leads_excluded=",".join(disc) or "none"),
            confidence=0.75,
        ))
    # Neutral descriptor: any lead with ST elevation at its threshold (contiguous or
    # not, primary or discordant). Never critical; matches the broad "ST elevation"
    # labels of the public datasets while the STEMI-grade alert stays strict.
    if any_elevated:
        fs.findings.append(Finding(
            code="st_elevation_any", group="ischemia",
            label_ru=f"Элевация ST в отведениях {', '.join(any_elevated)} (описательно)",
            label_uz=f"{', '.join(any_elevated)} tarmoqlarida ST elevatsiyasi (tavsifiy)",
            evidence=_ev(leads=",".join(any_elevated)), confidence=0.5,
        ))
    if len(early_repol) >= 2:
        fs.findings.append(Finding(
            code="early_repolarisation", group="other",
            label_ru=f"Ранняя реполяризация — вогнутая восходящая элевация ST, малая рядом с T ({', '.join(early_repol)}); вариант нормы",
            label_uz=f"Erta repolarizatsiya — T yonida kichik botiq ST elevatsiyasi ({', '.join(early_repol)}); norma varianti",
            evidence=_ev(leads=",".join(early_repol), j_t_ratio_max=er_ratio), confidence=0.6,
        ))
    if len(discordant) >= 2:
        fs.findings.append(Finding(
            code="secondary_st_t_change", group="ischemia",
            label_ru=f"Вторичные (дискордантные) ST-T изменения ({', '.join(discordant)})",
            label_uz=f"Ikkilamchi (diskordant) ST-T o'zgarishlar ({', '.join(discordant)})",
            evidence=_ev(leads=",".join(discordant),
                         context=",".join(sorted(fs.codes() & _SECONDARY_ST_CONTEXT))),
            confidence=0.6,
        ))


def _t_inversion_findings(m: Measurements, is_pediatric: bool, fs: FindingSet) -> None:
    inv = [l for l in m.morphology.inverted_t_leads if l != "aVR"]
    if not inv:
        return
    # Pediatric caveat (TZ §7.2): juvenile T-inversion in V1-V4 is NORMAL.
    if is_pediatric:
        inv = [l for l in inv if l not in _PEDIATRIC_TINV_OK]
        if not inv:
            fs.findings.append(Finding(
                code="juvenile_t_pattern", group="other",
                label_ru="Ювенильная инверсия T (V1–V4) — норма у детей",
                label_uz="Yuvenil T inversiyasi (V1–V4) — bolalarda norma",
                evidence=_ev(note="pediatric_normal_variant"), confidence=0.7,
            ))
            return
    if len(inv) >= 2:
        fs.findings.append(Finding(
            code="t_inversion", group="ischemia",
            label_ru=f"Инверсия зубца T ({', '.join(inv)})",
            label_uz=f"T to'lqin inversiyasi ({', '.join(inv)})",
            evidence=_ev(leads=",".join(inv)), confidence=0.65,
        ))


def _q_wave_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Pathological Q waves in >= 2 contiguous leads — possible prior infarction (D36).

    Q criteria are not interpretable with LBBB or pre-excitation (both create QS
    complexes and pseudo-Q waves), so those suppress the finding here; pacing is
    handled by the rhythm hierarchy in the pipeline.
    """
    qn = norms.get("q_wave")
    if not qn or not m.morphology.q_waves:
        return
    if fs.by_code("lbbb") is not None or fs.by_code("wpw_preexcitation") is not None:
        return
    patho = set()
    for lead, (q_ms, q_depth, r_amp, qs) in m.morphology.q_waves.items():
        if lead in ("aVR", "V1"):
            continue
        if lead in ("V2", "V3"):
            hit = qs or q_ms > qn["v2v3_q_ms_max"]
        else:
            hit = qs or (q_ms >= qn["q_ms_min"] and q_depth >= qn["q_depth_uv_min"]
                         and q_depth >= qn["q_depth_r_ratio_min"] * max(r_amp, 1.0))
        if hit:
            patho.add(lead)
    pairs = [(a, b) for a, b in _Q_CONTIGUOUS if a in patho and b in patho]
    if not pairs:
        return
    leads = [ld for ld in _Q_LEAD_ORDER if ld in {x for p in pairs for x in p}]
    walls = [(ru, uz) for ru, uz, group in _Q_WALLS if group & set(leads)]
    wall_ru = ", ".join(ru for ru, _ in walls) + (" стенки" if len(walls) > 1 else " стенка")
    wall_uz = ", ".join(uz for _, uz in walls) + (" devorlar" if len(walls) > 1 else " devor")
    leads_txt = ",".join(leads)
    fs.findings.append(Finding(
        code="pathological_q_waves", group="ischemia",
        label_ru=f"Патологические зубцы Q ({leads_txt}) — возможен перенесённый инфаркт "
                 f"({wall_ru})",
        label_uz=f"Patologik Q to'lqinlar ({leads_txt}) — o'tkazilgan infarkt ehtimoli "
                 f"({wall_uz})",
        evidence=_ev(leads=leads_txt, **{f"q_{ld}": m.morphology.q_waves[ld] for ld in leads}),
        confidence=0.6,
    ))


# Contiguous lead pairs and walls for pathological Q waves (D36).
_Q_CONTIGUOUS = [("I", "aVL"), ("II", "III"), ("III", "aVF"), ("II", "aVF"),
                 ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6")]
_Q_LEAD_ORDER = ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_Q_WALLS = [("нижняя", "pastki", {"II", "III", "aVF"}),
            ("передняя", "old", {"V1", "V2", "V3", "V4"}),
            ("боковая", "yon", {"I", "aVL", "V5", "V6"})]


_BRUGADA_J_UV = 200.0
_BRUGADA_T_UV = -50.0
_BRUGADA_RBBB_S_UV = -320.0


def _brugada_pattern(m: Measurements, fs: FindingSet) -> None:
    prof = m.morphology.right_precordial_st
    leads = [ld for ld, (j, j40, j80, t_min) in prof.items()
             if j >= _BRUGADA_J_UV and j40 < j and j80 < j40 and t_min <= _BRUGADA_T_UV]
    if not leads:
        return
    if any(s <= _BRUGADA_RBBB_S_UV for s in m.morphology.terminal_s_lateral.values()):
        return
    fs.findings.append(Finding(
        code="brugada_pattern", group="other",
        label_ru=f"Паттерн Бругада 1 типа? ({', '.join(leads)}: куполообразный подъём ST ≥ 2 мм с "
                 f"отрицательным T) — исключить БПНПГ и высокую установку электродов, консультация кардиолога",
        label_uz=f"Brugada 1-tur patterni? ({', '.join(leads)}: gumbazsimon ST ko'tarilishi ≥ 2 mm, manfiy T) — "
                 f"O'TBB va elektrodlarning baland o'rnatilishini istisno qiling, kardiolog maslahati",
        evidence=_ev(**{f"{ld}_j_j40_j80_tmin_uv": "/".join(str(v) for v in prof[ld]) for ld in leads}),
        # Natural-population PPV is unknown and low (the pattern is rare): lowest confidence.
        confidence=0.3,
    ))
