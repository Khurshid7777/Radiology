"""Bench metrics (TZ §11): signal agreement per trace, clinical intervals, calibration.

Signal metrics compare each true trace with the digitiser's trace of the same lead
after a time-origin search (``max_lag_s``): Pearson r, RMSE (mV) and SNR (dB) on
median-centred signals (a digitiser cannot know the paper's absolute baseline), and
the R-amplitude error (tallest positive deflection). Missing samples are excluded
and reported as lower coverage, never filled.

Clinical metrics follow SD-010: the *same* CardioLens delineator (Layers 2-3) runs on
(a) the truth as a perfect digitiser would deliver it (same layout windows) and
(b) the digitiser's output; the difference is the error the digitisation adds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from ecgscan.bench.cases import Case
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source, Trace

INTERVALS = ("rr_ms", "qrs_ms", "qt_ms", "pr_ms")


@dataclass
class TraceMetrics:
    lead: str
    kind: str
    r: float | None
    rmse_mv: float | None
    snr_db: float | None
    r_amp_err_mv: float | None
    lag_s: float | None
    coverage: float


@dataclass
class CaseMetrics:
    case_id: str
    slice: str
    digitizer: str
    status: str                  # ok | needs_confirmation | rejected | error
    error: str | None = None
    reject_reasons: list[str] = field(default_factory=list)
    elapsed_s: float | None = None
    traces: list[TraceMetrics] = field(default_factory=list)
    r_median: float | None = None
    rmse_median: float | None = None
    snr_median: float | None = None
    r_amp_err_median: float | None = None
    leads_missing: int = 0
    interval_err_ms: dict = field(default_factory=dict)   # name -> |Δ| ms or None
    unmeasured: list[str] = field(default_factory=list)   # measured on the truth, not on the output
    speed_ok: bool | None = None
    gain_ok: bool | None = None
    gain_rel_err: float | None = None
    px_per_mm_rel_err: float | None = None
    bad: bool | None = None      # any RR/QRS/QT error above the bad-interval threshold (or unmeasured)


def _centre(x: np.ndarray) -> np.ndarray:
    return x - np.nanmedian(x)


def compare_trace(truth: np.ndarray, t_start: float, out: np.ndarray, o_start: float, fs: float, *,
                  max_lag_s: float, min_overlap: float, snr_cap_db: float) -> dict | None:
    """Align ``out`` to ``truth`` in time and compare; None if they never overlap enough."""
    g0, o0 = int(round(t_start * fs)), int(round(o_start * fs))
    n_g, n_o = len(truth), len(out)
    max_lag = int(round(max_lag_s * fs))
    best = None
    for lag in range(-max_lag, max_lag + 1):
        a, b = max(g0, o0 + lag), min(g0 + n_g, o0 + lag + n_o)
        if b - a < min_overlap * n_g:
            continue
        g = truth[a - g0:b - g0]
        d = out[a - o0 - lag:b - o0 - lag]
        ok = np.isfinite(d)
        if ok.sum() < min_overlap * n_g or g[ok].std() == 0 or d[ok].std() == 0:
            continue
        r = float(np.corrcoef(g[ok], d[ok])[0, 1])
        if best is None or r > best[0]:
            best = (r, lag, g, d, ok)
    if best is None:
        return None
    r, lag, g, d, ok = best
    gc, dc = _centre(g[ok]), _centre(d[ok])
    err = gc - dc
    p_sig, p_err = float(np.sum(gc ** 2)), float(np.sum(err ** 2))
    snr = snr_cap_db if p_err == 0 else min(snr_cap_db, 10 * np.log10(max(p_sig, 1e-12) / p_err))
    return {"r": round(r, 4), "rmse_mv": round(float(np.sqrt(np.mean(err ** 2))), 4), "snr_db": round(float(snr), 2),
            "r_amp_err_mv": round(abs(float(gc.max()) - float(dc.max())), 4), "lag_s": round(lag / fs, 4),
            "coverage": round(float(ok.sum()) / n_g, 4)}


def truth_scan_result(case: Case) -> ScanResult:
    """The truth as a perfect digitiser would return it (same windows, exact samples)."""
    t = case.truth
    leads, rhythm = {}, []
    for tr in t.traces:
        trace = Trace.from_array(tr.samples_mv, start_s=tr.start_s, fs=t.fs, lead=tr.lead, quality=1.0)
        if tr.kind == "rhythm":
            rhythm.append(trace)
        else:
            leads[tr.lead] = trace
    return ScanResult(
        scan_id=str(uuid.uuid4()), status="ok", source=Source(original_sha256="0" * 64, pipeline_version="truth"),
        calibration=Calibration(paper_speed_mm_s=int(t.speed_mm_s), gain_mm_mv=int(t.gain_mm_mv),
                                speed_source="user", gain_source="user", confidence=1.0),
        layout=Layout(name=t.layout, confidence=1.0), sampling_rate_hz=t.fs, leads=leads, rhythm=rhythm,
        quality=QualitySummary(overall=1.0))


def clinical_measurements(result: ScanResult) -> dict[str, float | None]:
    """Heart rate and intervals by the CardioLens delineator (Layers 2-3)."""
    from cardiolens.config import load_app_config
    from cardiolens.layer2_preprocess import preprocess
    from cardiolens.layer3_measure import measure
    from ecgscan.contract.cardiolens_adapter import to_signal_representation

    cfg = load_app_config()
    pp = cfg.preprocessing
    sig = preprocess(to_signal_representation(result), mains_frequency_hz=cfg.mains_frequency_hz,
                     bandpass_low_hz=pp.bandpass_low_hz, bandpass_high_hz=pp.bandpass_high_hz,
                     target_sampling_rate_hz=pp.target_sampling_rate_hz)
    m = measure(sig)
    hr = m.heart_rate_bpm
    return {"heart_rate_bpm": hr, "rr_ms": 60000.0 / hr if hr else None, "qrs_ms": m.intervals.qrs_ms,
            "qt_ms": m.intervals.qt_ms, "pr_ms": m.intervals.pr_ms}


def _pick_output(result: ScanResult, lead: str, kind: str) -> Trace | None:
    same = result.leads.get(lead) if kind == "panel" else next((t for t in result.rhythm if t.lead == lead), None)
    if same is not None:
        return same
    other = next((t for t in result.rhythm if t.lead == lead), None) if kind == "panel" else result.leads.get(lead)
    return other


def _med(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 4) if vals else None


def score_case(case: Case, digitizer: str, run, reference: dict, cfg: dict) -> CaseMetrics:
    """All metrics of one digitiser on one case; ``reference`` = clinical measurements of the truth."""
    mcfg, acc = cfg["metrics"], cfg["acceptance"]
    cm = CaseMetrics(case_id=case.case_id, slice=case.slice, digitizer=digitizer, status="error",
                     error=run.error, elapsed_s=None if run.elapsed_s is None else round(run.elapsed_s, 3))
    res: ScanResult | None = run.result
    if res is None:
        cm.bad = True
        return cm
    cm.status = res.status
    cm.reject_reasons = list(res.reject_reasons)
    t = case.truth
    cal = res.calibration
    cm.speed_ok = cal.paper_speed_mm_s is not None and float(cal.paper_speed_mm_s) == float(t.speed_mm_s)
    gain_raw = (run.extra or {}).get("gain_raw") or cal.gain_mm_mv
    if gain_raw:
        cm.gain_rel_err = round(abs(float(gain_raw) - t.gain_mm_mv) / t.gain_mm_mv, 4)
    cm.gain_ok = cal.gain_mm_mv is not None and float(cal.gain_mm_mv) == float(t.gain_mm_mv)
    if cal.px_per_mm_x and case.meta.get("true_px_per_mm"):
        cm.px_per_mm_rel_err = round(abs(cal.px_per_mm_x - case.meta["true_px_per_mm"]) / case.meta["true_px_per_mm"], 4)
    if res.status == "rejected":
        return cm
    for tr in t.traces:
        out = _pick_output(res, tr.lead, tr.kind)
        if out is None:
            cm.leads_missing += 1
            cm.traces.append(TraceMetrics(tr.lead, tr.kind, None, None, None, None, None, 0.0))
            continue
        c = compare_trace(tr.samples_mv, tr.start_s, out.array(), out.start_s, t.fs,
                          max_lag_s=mcfg["max_lag_s"], min_overlap=mcfg["min_overlap"], snr_cap_db=mcfg["snr_cap_db"])
        if c is None:
            cm.traces.append(TraceMetrics(tr.lead, tr.kind, None, None, None, None, None, 0.0))
            continue
        cm.traces.append(TraceMetrics(tr.lead, tr.kind, c["r"], c["rmse_mv"], c["snr_db"], c["r_amp_err_mv"],
                                      c["lag_s"], c["coverage"]))
    cm.r_median = _med([x.r for x in cm.traces])
    cm.rmse_median = _med([x.rmse_mv for x in cm.traces])
    cm.snr_median = _med([x.snr_db for x in cm.traces])
    cm.r_amp_err_median = _med([x.r_amp_err_mv for x in cm.traces])
    try:
        dig = clinical_measurements(res)
    except Exception as exc:  # a broken signal may break the delineator: counts as unmeasured
        dig = {k: None for k in reference}
        cm.error = f"measure: {type(exc).__name__}: {exc}"[:200]
    for k in INTERVALS:
        ref, got = reference.get(k), dig.get(k)
        if ref is None:
            continue
        if got is None:
            cm.unmeasured.append(k)
            cm.interval_err_ms[k] = None
        else:
            cm.interval_err_ms[k] = round(abs(got - ref), 1)
    core = [cm.interval_err_ms.get(k) for k in ("rr_ms", "qrs_ms", "qt_ms") if k in cm.interval_err_ms]
    cm.bad = any(v is None or v > acc["bad_interval_ms"] for v in core) if core else True
    return cm
