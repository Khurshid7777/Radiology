"""End-to-end pipeline orchestration (TZ §3 layer flow).

Runs Layer 1 -> Layer 6 plus the cross-cutting safety scan, and assembles a
``PipelineResult`` carrying every intermediate artifact plus an audit log for
traceability (TZ §11.1, §12). Output rendering (JSON/SVG/graph/PDF) lives in
``cardiolens.outputs`` and consumes this result.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field
from pathlib import Path

from cardiolens.arrhythmia import (
    detect_pvc_beats,
    detect_ventricular_runs,
    pvc_findings,
    vt_run_finding,
)
from cardiolens.calibration import TemperatureScaler
from cardiolens.config import (
    AppConfig,
    configs_dir,
    load_app_config,
    load_rule_base,
    norms_for_patient,
)
from cardiolens.contracts.causal import CausalGraph
from cardiolens.contracts.features import Intervals, Measurements, Voltages
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.contracts.report import InterpretationReport
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer1_ingestion.checks import check_record
from cardiolens.layer1_ingestion.strip import LEADS_INCONSISTENT_NOTE, SCALE_CONFLICT_NOTE
from cardiolens.layer2_preprocess import assess_quality, preprocess
from cardiolens.layer2_preprocess.photo_quality import (
    align_panels,
    page_consistency,
    page_unreliable,
)
from cardiolens.layer3_measure import measure
from cardiolens.layer3_measure.pacing import (
    pacing_multilead_spike_times,
    pacing_spike_coupling,
    pacing_spike_ratio,
    pacing_spike_times,
)
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer4_findings.ml import apply_ml_evidence
from cardiolens.layer5_causal import analyze
from cardiolens.layer6_interpret import InterpretationInput, default_provider, interpret
from cardiolens.layer6_interpret.provider import LLMProvider
from cardiolens.safety import SafetyResult, scan_critical
from cardiolens.safety.critical import CriticalAlert
from cardiolens.serial import SerialComparison, compare


@dataclass
class PipelineResult:
    signal: SignalRepresentation
    measurements: Measurements
    findings: FindingSet
    causal_graph: CausalGraph
    safety: SafetyResult
    report: InterpretationReport
    config: AppConfig
    audit_log: list[str] = field(default_factory=list)
    serial: SerialComparison | None = None


# TZ 18.6: how far ours may differ from the recorder's printed values (D183).
DEVICE_TOLERANCE = {"hr_bpm": 5, "pr_ms": 15, "qrs_ms": 15, "qt_ms": 20}
_DEV_RU = {"hr_bpm": "ЧСС", "pr_ms": "PR", "qrs_ms": "QRS", "qt_ms": "QT"}
_DEV_UZ = {"hr_bpm": "YuUS", "pr_ms": "PR", "qrs_ms": "QRS", "qt_ms": "QT"}


def device_mismatches(m: Measurements, dev: dict[str, int]) -> list[tuple[str, float, int]]:
    """(field, ours, device) for each value both have that differs beyond the tolerance."""
    ours = {"hr_bpm": m.heart_rate_bpm, "pr_ms": m.intervals.pr_ms,
            "qrs_ms": m.intervals.qrs_ms, "qt_ms": m.intervals.qt_ms}
    return [(k, float(ours[k]), int(dev[k])) for k, tol in DEVICE_TOLERANCE.items()
            if ours.get(k) is not None and dev.get(k) is not None and abs(ours[k] - dev[k]) > tol]


def run_pipeline(
    sig: SignalRepresentation,
    *,
    mode: str | None = None,
    provider: LLMProvider | None = None,
    config: AppConfig | None = None,
    calibrator: TemperatureScaler | None = None,
    prior: PipelineResult | None = None,
) -> PipelineResult:
    """Run the full CardioLens pipeline on one signal."""
    cfg = config or load_app_config()
    rule_base = load_rule_base()
    # Confidence calibration (TZ §13): identity until a scaler is fitted/saved.
    if calibrator is None:
        calibrator = TemperatureScaler.load(configs_dir() / "calibrator.json")
    audit: list[str] = []
    fs_in = sig.sampling_rate_hz     # a digitised photo's px/mm x 25 mm/s, before resampling
    # A record that cannot be measured is refused here, whatever route it came by (D135).
    n_notes = len(sig.provenance.notes)
    sig = check_record(sig, cfg.ingestion)
    audit.extend(f"[L1] {n}" for n in sig.provenance.notes[n_notes:])

    # Pacing spikes are erased by the Layer-2 band-pass, so measure them on the raw signal.
    # Not on a digitised image (D114): at the pixel rate (~150 Hz) the steep QRS upstroke
    # is a one-sample step, the largest |diff| outlier, and it sits just before R, so the
    # coupling check passes too: 3 of 4 clean synthetic pages were read as paced.
    pacing_ratio = 0.0 if sig.provenance.digitized else pacing_spike_ratio(sig)
    spike_times = pacing_spike_times(sig) if pacing_ratio >= cfg.pacing.spike_ratio_min else []
    ml_spike_times = pacing_multilead_spike_times(sig, cfg.pacing.multilead_k,
                                                  cfg.pacing.multilead_min_leads)

    # Photo consistency on the digitised traces as read, before any filtering (D114); the
    # rows' time lag against the lead-II strip is then undone (D115).
    photo = page_consistency(sig)
    photo_aligned = None
    if photo is not None:
        sig = align_panels(sig)
        photo_aligned = page_consistency(sig)
        audit.append(f"[L1] page aligned on the lead-II strip: lag {photo['panel_lag_ms']} ms, "
                     f"Einthoven r {photo['einthoven_r']} -> {photo_aligned and photo_aligned['einthoven_r']}")

    # Layer 2 — preprocess + quality (TZ §5). preprocess() filters in place: work on a
    # copy so the caller's raw signal survives — a second run on the same object
    # would otherwise see the band-passed signal and lose the pacing spikes.
    pp = cfg.preprocessing
    raw = sig
    sig = preprocess(
        raw.model_copy(deep=True),
        mains_frequency_hz=cfg.mains_frequency_hz,
        bandpass_low_hz=pp.bandpass_low_hz,
        bandpass_high_hz=pp.bandpass_high_hz,
        target_sampling_rate_hz=pp.target_sampling_rate_hz,
    )
    # The QRS amplitude for low voltage on a wider band (D148): same resampling, so the
    # samples and R peaks line up; the 40-Hz pass shaves the QRS peaks (median 7 %).
    # Recorded signals only: on a digitised trace the wider band keeps the pixel steps
    # (real photos: +33 % "amplitude"), so photos stay on the measuring band.
    amp_sig = None
    if pp.amplitude_bandpass_high_hz and not raw.provenance.digitized:
        high = min(float(pp.amplitude_bandpass_high_hz), 0.45 * sig.sampling_rate_hz)
        if high > pp.bandpass_high_hz:
            amp_sig = preprocess(raw.model_copy(deep=True), mains_frequency_hz=cfg.mains_frequency_hz,
                                 bandpass_low_hz=pp.bandpass_low_hz, bandpass_high_hz=high,
                                 target_sampling_rate_hz=pp.target_sampling_rate_hz)
    quality, la_ra_reversed = assess_quality(sig, sqi_accept_threshold=pp.sqi_accept_threshold)
    quality_ok = _quality_ok(quality, pp.sqi_accept_threshold)
    audit.append(f"[L2] SQI mean-ok={quality_ok}, flags={quality.flags}")

    # Layer 3 — measurements (TZ §6).
    m = measure(sig, pacing_spikes_s=sorted(set(spike_times) | set(ml_spike_times)), amplitude_sig=amp_sig)
    audit.append(
        f"[L3] HR={m.heart_rate_bpm} PR={m.intervals.pr_ms} "
        f"QRS={m.intervals.qrs_ms} QTcB={m.intervals.qtc_bazett_ms}"
    )

    # Learnable layer (TZ §17): probabilities as extra evidence, never authority.
    model = _ml_model(cfg) if cfg.ml.enabled else None
    if model is not None:
        from cardiolens.ml.features import features_from_measurements

        m.ml_probabilities = model.predict_proba(features_from_measurements(m, sig.sampling_rate_hz))

    # Layer 4 — findings (TZ §7).
    norms = norms_for_patient(sig.patient.age_years, sig.patient.age_days)
    findings = detect_findings(sig, m, norms=norms, la_ra_reversed=la_ra_reversed)
    if model is not None:
        for line in apply_ml_evidence(findings, m, model.meta.get("suggest_classes")):
            audit.append(f"[ML] {line}")
    # Ventricular ectopy (TZ §7.1). VT (a §10 alert): runs of premature, abnormally
    # shaped beats on the 12-lead strip; isolated PVCs and bigeminy by the 12-lead
    # PVC rule (ported from claude/vibrant-curie-ro1zsz D25, D38, D40, D41).
    runs = detect_ventricular_runs(sig, m.r_peaks_samples or [], sig.sampling_rate_hz,
                                   recovered=m.r_peaks_recovered)
    if runs:
        findings.findings.append(vt_run_finding(runs))
        audit.append(f"[L4] ventricular run(s) (first beat, beats, rate): {runs}")
    pvc_idx = detect_pvc_beats(sig, m.r_peaks_samples or [], sig.sampling_rate_hz)
    if pvc_idx and not runs:
        findings.findings.extend(pvc_findings(pvc_idx, len(m.r_peaks_samples)))
        audit.append(f"[L4] 12-lead PVC rule: {len(pvc_idx)} PVC beat(s)")
        # AF read off the irregularity PVCs make (D106): the PVC rule needs a compensatory
        # pause, which AF lacks; without the RR intervals around the PVCs a sinus rhythm is
        # regular again, AF stays irregular.
        cv_left = _rr_cv_without(m.r_peaks_samples, pvc_idx)
        af_cv = norms.get("rhythm", {}).get("af_rr_cv_min", 0.10)
        if (AF_PVC_RECHECK and cv_left is not None and cv_left <= af_cv
                and "atrial_fibrillation" in findings.codes()):
            findings.findings = [f for f in findings.findings if f.code != "atrial_fibrillation"]
            audit.append(f"[L4] AF withdrawn: RR CV {cv_left:.3f} without the PVC intervals (D106)")
    # Paced rhythm (TZ §7.1): the spike must be time-locked to the QRS (an artifact
    # spike lands at random), or seen by many leads at once and QRS-locked (D75).
    r_times = [r / sig.sampling_rate_hz for r in (m.r_peaks_samples or [])]
    coupling = pacing_spike_coupling(spike_times, r_times)
    ml_coupling = pacing_spike_coupling(ml_spike_times, r_times)
    single = pacing_ratio >= cfg.pacing.spike_ratio_min and coupling >= cfg.pacing.coupling_min
    # Atrial pacing (D98): the spike precedes its QRS by the AV delay, beyond the coupling
    # window; beats whose detection was moved off such a spike are the paced ones.
    atrial = m.atrial_paced_beats / max(1, len(r_times))
    if single or ml_coupling >= cfg.pacing.multilead_coupling_min or atrial >= cfg.pacing.multilead_coupling_min:
        how_ru = (f"спайк ×{pacing_ratio}, связан с QRS в {coupling:.0%} ударов" if single else
                  f"спайк в ≥ {cfg.pacing.multilead_min_leads} отведениях, связан с QRS в {ml_coupling:.0%} ударов"
                  if ml_coupling >= cfg.pacing.multilead_coupling_min else
                  f"предсердная стимуляция: спайк за АВ-задержку до QRS в {atrial:.0%} ударов")
        findings.findings.append(Finding(
            code="paced_rhythm", group="rhythm",
            label_ru=f"Ритм электрокардиостимулятора ({how_ru})",
            label_uz="Kardiostimulyator ritmi",
            evidence=[Evidence(feature="pacing_spike_ratio", value=str(pacing_ratio)),
                      Evidence(feature="spike_qrs_coupling", value=str(coupling)),
                      Evidence(feature="multilead_spike_qrs_coupling", value=str(ml_coupling))],
            confidence=0.8))
        dropped = [f.code for f in findings.findings if f.code in _PACED_EXCLUDES]
        findings.findings = [f for f in findings.findings if f.code not in _PACED_EXCLUDES]
        audit.append(f"[L4] paced rhythm: spike_ratio={pacing_ratio} coupling={coupling} "
                     f"multilead_coupling={ml_coupling}; withheld {dropped}")
    # Pause / dropped beat (TZ §7.1, §10) — R-peak based.
    pc = cfg.pause
    if m.max_rr_ms is not None and (m.max_rr_ms >= pc.finding_ms
                                    or (m.pause_ratio or 0) >= pc.ratio):
        findings.findings.append(Finding(
            code="sinus_pause", group="rhythm",
            label_ru=f"Пауза ритма {m.max_rr_ms:.0f} мс (RR ×{m.pause_ratio})",
            label_uz=f"Ritm pauzasi {m.max_rr_ms:.0f} ms (RR ×{m.pause_ratio})",
            evidence=[Evidence(feature="max_rr_ms", value=f"{m.max_rr_ms:.0f}"),
                      Evidence(feature="pause_ratio", value=str(m.pause_ratio))],
            confidence=0.8, critical=m.max_rr_ms >= pc.critical_ms))
        audit.append(f"[L4] pause: max_rr={m.max_rr_ms} ms ratio={m.pause_ratio}")
    # Signal-organization gate (§5) — after all finding emission. If the QRS complexes
    # are not self-consistent, R-peaks (hence HR, intervals, rhythm) are unreliable:
    # withhold the findings and report poor quality; mode B escalates.
    tc = m.qrs_template_consistency
    pc = m.qrs_partner_consistency
    # Too few complexes to judge (fewer than 3 away from the edges) is not "consistent":
    # band-limited noise with 3 such "beats" read as AF and HR 32 with SQI 0.8 (D135).
    # A device screen's ~2-s window (D122) holds two beats by design and is read from
    # them; the rule is for records long enough to hold more (merge of the scanner branch).
    page_edges = [0.0, *(sig.provenance.discontinuities_s or []), sig.duration_s]
    longest_page = max(b - a for a, b in itertools.pairwise(page_edges))
    unjudged = tc is None and len(m.r_peaks_samples or []) > 0 and longest_page >= SHORT_WINDOW_S
    if unjudged or (tc is not None and tc < cfg.quality.beat_consistency_min
                    and not (pc is not None and pc >= cfg.quality.beat_partner_min)):
        why = ("too few QRS complexes to check their consistency" if unjudged
               else f"QRS consistency {tc} < {cfg.quality.beat_consistency_min}")
        findings.findings = [Finding(
            code="poor_quality_signal", group="quality",
            label_ru=("Сигнал непригоден для надёжной интерпретации ("
                      + ("слишком мало комплексов QRS для проверки)" if unjudged
                         else f"консистентность QRS {tc} < {cfg.quality.beat_consistency_min})")),
            label_uz=("Signal ishonchli talqin uchun yaroqsiz ("
                      + ("tekshirish uchun QRS komplekslari juda kam)" if unjudged
                         else f"QRS konsistentligi {tc})")),
            evidence=[Evidence(feature="qrs_template_consistency", value=str(tc))],
            confidence=0.9)]
        quality_ok = False
        audit.append(f"[L2] signal disorganized ({why}) — R-peak findings withheld")
    # Digitised page: the final beats disagree with the page's column detector (D92 of the
    # scanner branch, now D130).
    if m.page_beat_agreement is False:
        quality_ok = False
        audit.append("[L3] beats disagree with the page columns (D130) — rhythm uncertain")
    # No ECG at all (D108): neither a rhythm nor an alert can be read from a flat line or
    # baseline wander; the disorganisation gate above needs beats to judge.
    hf_uv = _hf_amplitude_uv(sig)
    no_ecg = hf_uv < cfg.quality.min_qrs_hf_uv
    # A units error (D119): no ECG reaches 100 mV; every voltage, ST and alert read off
    # such a file is off by the same factor.
    bad_units = hf_uv > cfg.quality.max_qrs_hf_uv
    if bad_units:
        findings.findings = [Finding(
            code="amplitude_implausible", group="quality",
            label_ru=f"Амплитуда сигнала {hf_uv / 1000:.0f} мВ невозможна для ЭКГ — проверьте единицы файла (мкВ / мВ)",
            label_uz=f"Signal amplitudasi {hf_uv / 1000:.0f} mV EKG uchun imkonsiz — fayl birliklarini tekshiring (µV / mV)",
            evidence=[Evidence(feature="hf_peak_to_peak_uv", value=f"{hf_uv:.0f}")],
            confidence=0.95)]
        quality_ok = False
        audit.append(f"[L2] amplitude {hf_uv:.0f} µV is not an ECG (units?) — all findings and alerts withheld")
    # A digitised photo whose page geometry was not recovered (D114): leads shifted in
    # time widen the QRS and break the RR, so nothing read from it is trusted.
    # The grid pitch and the calibration pulses disagree (D128): the time scale is not
    # known, so neither is the rate — Einthoven holds on a uniformly stretched record.
    scale_conflict = sig.provenance.digitized and SCALE_CONFLICT_NOTE in sig.provenance.notes
    # The scanner's own lead checks disagree and the CardioLens readers could not read it
    # better (D181: ECG-Image-Kit pages read with a known scale, rate 124 as 89, no gate).
    leads_inconsistent = sig.provenance.digitized and LEADS_INCONSISTENT_NOTE in sig.provenance.notes
    # Below the digital minimum rate (D135: 50 Hz read QRS 178 ms, a false LBBB) a photo is
    # too small to measure; it is gated with the retake hint rather than refused.
    coarse_photo = sig.provenance.digitized and fs_in < cfg.ingestion.min_sampling_rate_hz
    bad_photo = not no_ecg and not bad_units and (
        scale_conflict or leads_inconsistent or coarse_photo or page_unreliable(photo, photo_aligned, cfg.quality.photo_max_panel_lag_ms,
                                                          cfg.quality.photo_min_einthoven_r))
    if bad_photo:
        lag = photo["panel_lag_ms"] if photo else None
        ein = (photo_aligned or photo)["einthoven_r"] if photo else None
        why_ru, why_uz = (("масштаб не определён: сетка и калибровочный импульс расходятся",
                           "masshtab aniqlanmadi: to'r va kalibrlash impulsi mos emas") if scale_conflict else
                          ("отведения на снимке не согласуются между собой",
                           "suratdagi tarmoqlar o'zaro mos emas") if leads_inconsistent else
                          ("снимок слишком мелкий для измерения", "surat o'lchash uchun juda kichik")
                          if coarse_photo and not (photo and page_unreliable(
                              photo, photo_aligned, cfg.quality.photo_max_panel_lag_ms,
                              cfg.quality.photo_min_einthoven_r)) else
                          ("отведения не совпадают по времени", "tarmoqlar vaqt bo'yicha mos emas"))
        # A digitised record's rate is the photo's px/mm x 25 mm/s: below ~6 px/mm a QRS
        # is a few pixels wide and the rows cannot be told apart where they cross (D126).
        # Messengers shrink photos to ~1280 px; sent as a file the photo keeps its size.
        px_mm = fs_in / 25.0
        low_res = px_mm < PHOTO_LOW_RES_PX_PER_MM
        on_screen = any(n.startswith("device screen photo") for n in sig.provenance.notes)
        what_ru, what_uz = ("экран", "ekran") if on_screen else ("лист", "varaq")
        findings.findings = [Finding(
            code="photo_unreliable", group="quality",
            label_ru=f"Снимок ЭКГ не распознан надёжно ({why_ru}) — "
                     + (f"снимок мелкий (~{px_mm:.1f} пикс/мм, нужно от {PHOTO_LOW_RES_PX_PER_MM:.0f}): "
                        f"снимите ближе, чтобы {what_ru} занял весь кадр, и отправьте как файл, а не как фото"
                        if low_res else "переснимите лист ровно, без перекоса и тени"),
            label_uz=f"EKG surati ishonchli o'qilmadi ({why_uz}) — "
                     + (f"surat kichik (~{px_mm:.1f} piksel/mm, kamida {PHOTO_LOW_RES_PX_PER_MM:.0f} kerak): "
                        f"yaqinroqdan, {what_uz} butun kadrni egallaydigan qilib oling va rasm emas, fayl sifatida yuboring"
                        if low_res else "varaqni tekis, qiyshaymasdan va soyasiz qayta suratga oling"),
            evidence=[Evidence(feature="panel_lag_ms", value=f"{lag:.0f}" if lag is not None else "—"),
                      Evidence(feature="einthoven_r", value=f"{ein:.2f}" if ein is not None else "—"),
                      Evidence(feature="px_per_mm", value=f"{px_mm:.1f}"),
                      Evidence(feature="scale_conflict", value="yes" if scale_conflict else "no")],
            confidence=0.9)]
        quality_ok = False
        audit.append(f"[L2] digitised page inconsistent (lag {lag} ms, Einthoven r {ein}, scale conflict "
                     f"{scale_conflict}) — findings and alerts withheld")
    # A ~2-s window with one QRS (D122): no RR to measure, but the neighbouring beats lie
    # outside the window (or at its undetected edges), so the rate is below
    # 60 / (the larger gap to an edge - 0.5 s). A severe
    # bradycardia on a device preview would otherwise pass as "no rhythm findings".
    # Pages joined from several photos (D123) are separate windows: the gaps are measured
    # within the beat's own page.
    edges = [0.0, *(sig.provenance.discontinuities_s or []), sig.duration_s]
    pages = list(itertools.pairwise(edges))
    longest = max(b - a for a, b in pages)
    n_rp = len(m.r_peaks_samples or [])
    if not (no_ecg or bad_units or bad_photo) and longest < SHORT_WINDOW_S and n_rp <= 1 and not m.heart_rate_bpm:
        # Beats within ~0.5 s of an edge are not detected (truncated), so the gap is shortened by that.
        if n_rp == 1:
            t_beat = m.r_peaks_samples[0] / sig.sampling_rate_hz
            a, b = next(((a, b) for a, b in pages if a <= t_beat <= b), (0.0, sig.duration_s))
            gap = max(t_beat - a, b - t_beat) - SHORT_WINDOW_EDGE_S
        else:
            gap = longest - 2 * SHORT_WINDOW_EDGE_S
        bound = 60.0 / max(0.5, gap)
        seen = "виден один комплекс" if n_rp == 1 else "нет ни одного комплекса"
        seen_uz = "bitta kompleks" if n_rp == 1 else "birorta ham kompleks yo'q"
        findings.findings.append(Finding(
            code="rate_unmeasurable_short", group="quality",
            label_ru=f"В окне {longest:.1f} с {seen}: ЧСС не измерить, вероятно ниже "
                     f"{bound:.0f}/мин (возможна брадикардия) — запишите не менее 10 с",
            label_uz=f"{longest:.1f} s oynada {seen_uz}: YUT o'lchab bo'lmaydi, ehtimol "
                     f"{bound:.0f}/daq dan past (bradikardiya bo'lishi mumkin) — kamida 10 s yozing",
            evidence=[Evidence(feature="max_rate_bpm", value=f"{bound:.0f}")], confidence=0.8))
    # A messenger-compressed photo read in the scanner's coarse mode (D170): at ~3-4 px/mm
    # one pixel is 10-13 ms, so the rate and the rhythm hold (RR within ~15 ms) while the
    # QRS and QT widths, the ST segments and the voltages do not. They are not measured,
    # the findings that rest on them are withheld, and the report says why.
    coarse = next((n for n in sig.provenance.notes if n.startswith(COARSE_NOTE_PREFIX)), None)
    if coarse and not (no_ecg or bad_units or bad_photo):
        px_mm = coarse.partition("px_per_mm=")[2].split()[0] if "px_per_mm=" in coarse else "?"
        m.intervals, m.voltages, m.sokolow_lyon_uv = Intervals(), Voltages(), None
        dropped = sorted(f.code for f in findings.findings if f.group in COARSE_WITHHELD_GROUPS)
        findings.findings = [f for f in findings.findings if f.group not in COARSE_WITHHELD_GROUPS]
        # ST is re-read with the thresholds raised by one millimetre (~3 px at 3 px/mm): a 2-mm
        # elevation or depression is six pixels and shows; a 1-mm one is noise. What passes is
        # said to be approximate; the alert it raises is marked "check against the record".
        from cardiolens.layer4_findings.rules import _st_findings

        st_fs = FindingSet(findings=list(findings.findings))
        _st_findings(m, {**norms, "st_uv": coarse_st_thresholds(norms.get("st_uv", {}))}, st_fs,
                     patient=sig.patient)
        for f in st_fs.findings:
            if f.group == "ischemia":
                f.label_ru += " — по сжатому снимку, ориентировочно"
                f.label_uz += " — siqilgan surat bo'yicha, taxminiy"
                f.confidence = round(f.confidence * 0.8, 2)
                findings.findings.append(f)
        findings.findings.append(Finding(
            code="photo_coarse", group="quality",
            label_ru=f"Сжатый снимок (~{px_mm} пикс/мм, нужно от 8): ЧСС и ритм ориентировочны, "
                     "интервалы, ST и вольтаж не измеряются — для полного заключения нужен оригинал с камеры",
            label_uz=f"Siqilgan surat (~{px_mm} piksel/mm, kamida 8 kerak): YUT va ritm taxminiy, "
                     "intervallar, ST va voltaj o'lchanmaydi — to'liq xulosa uchun kameradan asl surat kerak",
            evidence=[Evidence(feature="px_per_mm", value=px_mm)], confidence=0.9))
        quality_ok = False       # mode B escalates; any alert is marked "check against the record"
        audit.append(f"[L2] coarse photo ({px_mm} px/mm, D170): intervals, ST and voltages not measured; "
                     f"withheld {dropped}")
    # A device-screen photo (D120): the scale is assumed, only the leads on the screen
    # exist and the window is ~2 s. Findings stay, the limitation is said in the report.
    screen = next((n for n in sig.provenance.notes if n.startswith("device screen photo")), None)
    if screen and not (no_ecg or bad_units or bad_photo):
        findings.findings.append(Finding(
            code="screen_photo_limited", group="quality",
            label_ru="Фото экрана прибора: масштаб принят по умолчанию (25 мм/с, 10 мм/мВ), видны не все отведения, "
                     "запись короткая — амплитуды и ST ориентировочны; для заключения нужна полная запись 12 отведений",
            label_uz="Qurilma ekrani surati: masshtab taxminiy (25 mm/s, 10 mm/mV), barcha tarmoqlar ko'rinmaydi, "
                     "yozuv qisqa — amplituda va ST taxminiy; xulosa uchun to'liq 12 tarmoqli yozuv kerak",
            evidence=[Evidence(feature="screen", value=screen)], confidence=0.9))
    if no_ecg:
        findings.findings = [Finding(
            code="no_ecg_signal", group="quality",
            label_ru="ЭКГ-сигнал не обнаружен (нет комплексов QRS ни в одном отведении) — проверьте электроды"
                     + (_UNITS_HINT_RU if sig.source in _FILE_SOURCES else ""),
            label_uz="EKG signali aniqlanmadi (hech bir tarmoqda QRS yo'q) — elektrodlarni tekshiring"
                     + (_UNITS_HINT_UZ if sig.source in _FILE_SOURCES else ""),
            evidence=[Evidence(feature="hf_peak_to_peak_uv", value=f"{hf_uv:.0f}")],
            confidence=0.95)]
        quality_ok = False
        audit.append("[L2] no ECG signal (8-40 Hz below threshold in every lead) — all findings and alerts withheld")
    # The recorder's own printed measurements against ours (scanner S3, TZ 18.6, D183). Advisory:
    # a mis-read digit in the header must not withhold our numbers, so it never gates.
    dev = sig.provenance.device_measurements
    if dev and not (no_ecg or bad_units or bad_photo):
        off = device_mismatches(m, dev)
        if off:
            findings.findings.append(Finding(
                code="device_mismatch", group="quality",
                label_ru="Наши измерения расходятся с распечатанными аппаратом: "
                         + "; ".join(f"{_DEV_RU[k]} {o:.0f} (аппарат {d})" for k, o, d in off)
                         + " — сверьте с записью",
                label_uz="O'lchovlarimiz apparat chop etganidan farq qiladi: "
                         + "; ".join(f"{_DEV_UZ[k]} {o:.0f} (apparat {d})" for k, o, d in off)
                         + " — yozuv bilan solishtiring",
                evidence=[Evidence(feature=f"device_{k}", value=f"ours {o:.0f} / device {d}") for k, o, d in off],
                confidence=0.6))
            audit.append(f"[L2] device mismatch (D183): {[(k, round(o), d) for k, o, d in off]}")
    # Apply confidence calibration to each finding (TZ §13).
    for f in findings.findings:
        f.confidence = round(calibrator.apply(f.confidence), 3)
    audit.append(
        f"[L4] findings={[f.code for f in findings.findings]} "
        f"(calibrated T={calibrator.temperature})"
    )

    # Serial comparison against a prior ECG (TZ §8.4).
    serial: SerialComparison | None = None
    if prior is not None:
        serial = compare(findings, m, prior.findings, prior.measurements,
                         prior_id=sig.context.prior_ecg_id)
        audit.append(f"[serial] deltas={[d.code for d in serial.deltas]} "
                     f"qtcΔ={serial.qtc_delta_ms}")

    # Safety scan (TZ §10) — before interpretation, independent of LLM.
    # Two policies meet here. A record that is not an ECG (D108), off by a units factor
    # (D119) or a photo whose geometry or scale was not recovered (D114, D128) yields no
    # alert: nothing read off it is a measurement. Any other quality failure keeps its
    # alerts (recall first, TZ §10) and says so (scanner branch D97, now D135).
    hard_gate = no_ecg or bad_photo or bad_units
    safety = scan_critical(findings, Measurements() if hard_gate else m)
    if not quality_ok and not hard_gate:
        for a in safety.alerts:
            a.unverified = True
            a.label_ru += " — сигнал низкого качества, проверить по записи"
            a.label_uz += " — signal sifati past, yozuv bo'yicha tekshiring"
    # New critical dynamics on serial comparison force escalation (TZ §8.4, §10).
    if serial is not None:
        for d in serial.deltas:
            if d.critical:
                safety.alerts.append(CriticalAlert(
                    code=f"serial_{d.code}", label_ru=d.label_ru, label_uz=d.label_uz,
                    reason="significant serial change", log_id=f"serial-{d.code}"))
                safety.audit_log.append(f"[serial] CRITICAL new/changed {d.code}")
    audit.extend(safety.audit_log)

    # Layer 5 — causal analysis (TZ §8).
    causal = analyze(findings, rule_base, sig.context)
    # Serial dynamics strengthen confidence (TZ §8.4).
    if serial is not None and serial.has_significant_change:
        causal.confidence = round(min(1.0, causal.confidence + 0.05), 3)
    audit.append(
        f"[L5] rules={causal.activated_rules} critical={causal.critical_flags} "
        f"implausible={causal.implausible_flags} conf={causal.confidence}"
    )

    # A signal digitised from a photo is no more certain than its digitisation
    # (ECG scanner SD-015): the overall confidence is capped by the scanner's.
    prov = sig.provenance
    if prov.digitized and prov.confidence is not None and causal.confidence > prov.confidence:
        audit.append(f"[L5] confidence {causal.confidence} capped by digitisation {prov.confidence}")
        causal.confidence = round(float(prov.confidence), 3)

    # Layer 6 — interpretation with mode policy (TZ §9, §2).
    requested_mode = mode or cfg.mode
    data = InterpretationInput(
        findings=findings, causal_graph=causal, measurements=m,
        patient=sig.patient, context=sig.context, mode=requested_mode,
        critical_labels=[(a.label_ru, a.label_uz) for a in safety.alerts],
        disclaimer_ru=cfg.disclaimers.ru, disclaimer_uz=cfg.disclaimers.uz,
    )
    report = interpret(
        data, safety, provider=provider or default_provider(llm=cfg.llm),
        min_overall=cfg.confidence.autonomous_min_overall, quality_ok=quality_ok,
        digitized=prov.digitized,
    )
    audit.append(f"[L6] provider={report.mode} escalation={report.escalation_reason}")

    return PipelineResult(
        signal=sig, measurements=m, findings=findings, causal_graph=causal,
        safety=safety, report=report, config=cfg, audit_log=audit, serial=serial,
    )


# A file in mV read as µV is 1000x too small and looks like no ECG at all (D119).
PHOTO_LOW_RES_PX_PER_MM = 6.0   # D126: an unreliable photo below this asks for a closer shot sent as a file
# The scanner's coarse mode (D170) marks its signal with this provenance note:
# "coarse: px_per_mm=3.1". The finding groups that need a fine time or voltage scale.
COARSE_NOTE_PREFIX = "coarse:"
COARSE_WITHHELD_GROUPS = frozenset({"interval", "ischemia", "hypertrophy"})
COARSE_ST_MARGIN_UV = 100.0      # one millimetre on top of every ST threshold (provisional, D170)


def coarse_st_thresholds(st: dict, margin: float = COARSE_ST_MARGIN_UV) -> dict:
    """The ST thresholds for a compressed photo (D170): every significant level one
    millimetre higher, the 0.5-1 mm descriptor off (below what 3 px/mm resolves)."""
    out = dict(st)
    for key in ("elevation_significant", "depression_significant", "any_elevation_uv",
                "proportional_discordant_min_uv"):
        if out.get(key) is not None:
            out[key] = float(out[key]) + margin
    if isinstance(out.get("elevation_v2_v3"), dict):
        out["elevation_v2_v3"] = {k: float(v) + margin for k, v in out["elevation_v2_v3"].items()}
    out.pop("depression_minor", None)
    out.setdefault("any_elevation_uv", out.get("elevation_significant"))
    return out
SHORT_WINDOW_S = 4.0     # D122: below this, one visible QRS bounds the rate instead of measuring it
SHORT_WINDOW_EDGE_S = 0.5
_FILE_SOURCES = {"csv", "edf"}
_UNITS_HINT_RU = "; для файла — проверьте единицы (мВ вместо мкВ)"
_UNITS_HINT_UZ = "; fayl uchun birliklarni tekshiring (µV o'rniga mV)"


def _hf_amplitude_uv(sig: SignalRepresentation) -> float:
    """Largest 8-40 Hz peak-to-peak (0.5-99.5 percentile) over the leads, µV (D108)."""
    import numpy as np
    from scipy.signal import butter, filtfilt

    fs = sig.sampling_rate_hz
    x = np.nan_to_num(np.asarray(sig.signal, dtype=np.float64))
    if x.shape[-1] < int(fs):
        return 0.0
    b, a = butter(2, [8.0 / (fs / 2), min(40.0, fs / 2 - 1) / (fs / 2)], btype="band")
    y = filtfilt(b, a, x, axis=-1)
    return float(np.max(np.percentile(y, 99.5, axis=-1) - np.percentile(y, 0.5, axis=-1)))


AF_PVC_RECHECK = True   # D106: accepted on the held-out check (fold 10 + Georgia g6-g11)


def _rr_cv_without(rpeaks, pvc_idx) -> float | None:
    """RR coefficient of variation leaving out the intervals into and out of each PVC."""
    import numpy as np

    rr = np.diff(np.asarray(rpeaks, dtype=float))
    drop = {k for i in pvc_idx for k in (i - 1, i)}
    keep = np.array([v for k, v in enumerate(rr) if k not in drop])
    if keep.size < 4:
        return None
    return float(np.std(keep) / (np.mean(keep) + 1e-9))


# A paced QRS is wide with secondary ST-T and a left axis by construction: these are not
# separate diagnoses under a pacemaker (ported from claude/vibrant-curie-ro1zsz).
# Nor is AF by the P/RR rule (D91): paced beats have no P and the pacer sets the RR;
# on 294 PTB-XL paced records it fired 55 times, once with AF in the report.
_PACED_EXCLUDES = ("lbbb", "rbbb", "wide_qrs_nonspecific", "lvh", "lvh_probable",
                   "first_degree_av_block", "sinus_rhythm", "left_axis_deviation", "st_depression",
                   "t_inversion", "secondary_st_t_change", "pathological_q_waves", "brugada_pattern",
                   "atrial_fibrillation")


@functools.lru_cache(maxsize=2)
def _load_ml(path: str):
    from cardiolens.ml.model import load_model

    return load_model(path)


def _ml_path(cfg: AppConfig) -> Path:
    p = Path(cfg.ml.model_path)
    return p if p.is_absolute() else configs_dir().parent / p


def _ml_model(cfg: AppConfig):
    """The learnable-layer model named in the config, or None (rules-only)."""
    return _load_ml(str(_ml_path(cfg)))


def ml_status(cfg: AppConfig) -> dict:
    """What /health reports about the learnable layer (D167): ``loaded`` when the model file
    is present and usable, ``absent`` when the pipeline runs on rules alone."""
    p = _ml_path(cfg)
    model = _ml_model(cfg) if cfg.ml.enabled else None
    return {"enabled": cfg.ml.enabled, "model": "loaded" if model is not None else "absent",
            "path": str(p), "file_present": p.exists(),
            "classes": sorted(model.classes) if model is not None and hasattr(model, "classes") else []}


def _quality_ok(quality, threshold: float) -> bool:
    sqi = quality.per_lead_sqi
    if not sqi:
        return False
    good = sum(1 for s in sqi if s >= threshold)
    return good >= 0.7 * len(sqi)  # >=70% of leads acceptable
