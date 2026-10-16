"""Per-beat classification for ventricular ectopy (TZ §7.1).

Rate-based rhythm findings (brady/tachy, AF) ride reliable R-peaks; ectopic-beat
findings need per-beat morphology, which a single median interval can't express.
This module classifies each beat as N (normal), S (supraventricular ectopic) or
V (ventricular ectopic) from scale-invariant features, using a transparent LDA
whose coefficients live in ``configs/beat_lda.json`` (trained offline on MIT-BIH;
see ``cardiolens.train_beats``). Like the confidence calibrator, it degrades to a
no-op when the model file is absent.

Only ventricular ectopy (V) is reported: on an inter-patient MIT-BIH split it
reaches ~0.84 sensitivity / 0.99 PPV, whereas the supraventricular class is not
reliably separable from normal with these features (documented in DECISIONS.md).
Features are scale/rate-invariant (RR ratios, QRS width in ms, amplitude relative
to the record's own beats, morphology correlation to the dominant beat) so a model
trained at 360 Hz in mV transfers to the pipeline's 500 Hz µV lead II.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# QRS half-window for morphology features (~70 ms each side).
_HALF_MS = 70.0
# Number of normalized QRS-shape samples (scale/rate-invariant morphology).
# Higher K doesn't raise VEB sensitivity (a feature-family ceiling ~0.67) but
# sharpens precision: 10->20 lifts VEB PPV 0.86->0.89 at the same recall.
_SHAPE_K = 20

FEATURES = (["pre_rr_ratio", "post_rr_ratio", "local_rr_ms",
             "qrs_width_ms", "amp_ratio", "template_corr", "prematurity", "comp_ratio"]
            + [f"shape_{i}" for i in range(_SHAPE_K)])


def extract_beat_features(lead: np.ndarray, rpeaks, fs: float) -> tuple[np.ndarray, list[int]]:
    """Scale-invariant per-beat features from one (filtered) lead in µV.

    Returns the feature matrix and the beat indices it covers (interior beats
    with a full QRS window). The morphology template is the median of ALL beat
    windows — label-free, so runtime matches training. See module docstring.
    """
    lead = np.asarray(lead, dtype=float)
    rp = np.asarray(rpeaks, dtype=int)
    n = len(rp)
    half = int(_HALF_MS / 1000.0 * fs)
    if n < 4 or half < 3:
        return np.empty((0, len(FEATURES))), []

    rr = np.diff(rp) / fs * 1000.0  # ms, length n-1
    segs = [lead[r - half:r + half] for r in rp if r - half >= 0 and r + half < len(lead)]
    if len(segs) < 10:
        return np.empty((0, len(FEATURES))), []
    tmpl = np.median(np.vstack(segs), axis=0)
    tmpl_amp = float(np.max(tmpl) - np.min(tmpl)) or 1.0

    rows, idx = [], []
    for i in range(1, n - 1):
        r = rp[i]
        if r - half < 0 or r + half >= len(lead):
            continue
        pre, post = rr[i - 1], rr[i]
        lo = max(0, i - 6)
        loc = float(np.median(rr[lo:i + 6])) if i + 6 <= len(rr) else float(np.median(rr))
        loc = loc or 1.0
        seg = lead[r - half:r + half]
        base = np.median(seg)
        amp = (float(np.max(seg) - np.min(seg))) / (tmpl_amp + 1e-9)
        thr = 0.2 * np.max(np.abs(seg - base))
        above = np.where(np.abs(seg - base) >= thr)[0]
        width = (above[-1] - above[0]) / fs * 1000.0 if len(above) >= 2 else 0.0
        denom = (np.std(seg) * np.std(tmpl))
        corr = float(np.mean((seg - seg.mean()) * (tmpl - tmpl.mean())) / denom) if denom else 0.0
        # Normalized QRS shape (baseline-removed, unit-amplitude, resampled to K):
        # scale/rate-invariant morphology that separates wide ectopic beats.
        norm = (seg - base)
        peak = np.max(np.abs(norm)) or 1.0
        shape = np.interp(np.linspace(0, len(norm)-1, _SHAPE_K),
                          np.arange(len(norm)), norm / peak)
        rows.append([pre / loc, post / loc, loc, width, amp, corr,
                     (loc - pre) / loc, post / loc, *shape.tolist()])
        idx.append(i)
    return np.asarray(rows), idx


@dataclass
class BeatClassifier:
    """Transparent LDA over beat features; loads coefficients from JSON."""

    classes: list[str] = field(default_factory=list)
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    means: dict[str, np.ndarray] = field(default_factory=dict)
    inv_cov: np.ndarray | None = None
    log_priors: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    @classmethod
    def load(cls, path: str | Path) -> BeatClassifier:
        path = Path(path)
        if not path.exists():
            return cls()  # no-op until a model is trained (mirrors the calibrator)
        d = json.loads(path.read_text())
        import math
        return cls(
            classes=list(d["classes"]),
            mu=np.asarray(d["mu"], float), sd=np.asarray(d["sd"], float),
            means={c: np.asarray(v, float) for c, v in d["means"].items()},
            inv_cov=np.asarray(d["inv_cov"], float),
            log_priors={c: math.log(p) for c, p in d["priors"].items()},
            fitted=True,
        )

    def predict(self, X: np.ndarray) -> list[str]:
        if not self.fitted or len(X) == 0:
            return ["N"] * len(X)
        Xz = (X - self.mu) / self.sd
        out = []
        for x in Xz:
            best, bs = "N", -np.inf
            for c in self.classes:
                d = x - self.means[c]
                score = -0.5 * d @ self.inv_cov @ d + self.log_priors[c]
                if score > bs:
                    bs, best = score, c
            out.append(best)
        return out


def ectopy_findings(summary: EctopySummary, *, vt_only: bool = False):
    """Build ventricular-ectopy findings from a beat-classification summary (§7.1).

    Reported only when ≥2 ventricular beats are classified (a single beat is
    within the per-beat error), so an incidental misclassification on a short
    strip doesn't raise a finding.
    """
    from cardiolens.contracts.findings import Evidence, Finding

    out = []
    if summary.n_ventricular < 2:
        return out
    ev = [Evidence(feature="pvc_count", value=str(summary.n_ventricular)),
          Evidence(feature="beats", value=str(summary.n_beats)),
          Evidence(feature="burden", value=f"{summary.burden:.0%}")]
    # A fast run of >=4 consecutive ventricular beats is ventricular tachycardia
    # — a §10 must-not-miss; takes precedence. Gated for zero false positives on
    # MIT-BIH DS2 windows (see summarize_ectopy / DECISIONS.md D15b).
    if summary.vt:
        out.append(Finding(
            code="ventricular_tachycardia", group="rhythm",
            label_ru=f"Желудочковая тахикардия (пробежка {summary.max_vt_run} ЖЭ подряд)",
            label_uz=f"Ventrikulyar taxikardiya ({summary.max_vt_run} ta ketma-ket VES)",
            evidence=[*ev, Evidence(feature="run", value=str(summary.max_vt_run))],
            confidence=0.8, critical=True,
        ))
        return out
    if vt_only:
        # D25: isolated-PVC and bigeminy calls come from detect_pvc_beats; the
        # Holter-trained LDA over-calls them on 10-s 12-lead records.
        return out
    if summary.bigeminy:
        out.append(Finding(
            code="ventricular_bigeminy", group="rhythm",
            label_ru=f"Желудочковая бигеминия (ЖЭ {summary.burden:.0%} ударов)",
            label_uz=f"Ventrikulyar bigeminiya (VES {summary.burden:.0%})",
            evidence=ev, confidence=0.7,
        ))
    else:
        code = "ventricular_couplet" if summary.max_vt_run == 2 else "ventricular_ectopy"
        label_ru = ("Парная желудочковая экстрасистолия (куплет"
                    if code == "ventricular_couplet" else "Желудочковая экстрасистолия")
        label_uz = ("Juft ventrikulyar ekstrasistoliya (kuplet"
                    if code == "ventricular_couplet" else "Ventrikulyar ekstrasistoliya")
        suffix = ")" if code == "ventricular_couplet" else ""
        out.append(Finding(
            code=code, group="rhythm",
            label_ru=f"{label_ru}{suffix} — {summary.n_ventricular} ЖЭ / "
                     f"{summary.n_beats} уд., {summary.burden:.0%}",
            label_uz=f"{label_uz}{suffix} — {summary.n_ventricular} VES / "
                     f"{summary.n_beats}, {summary.burden:.0%}",
            evidence=ev, confidence=0.75,
        ))
    return out


@dataclass
class EctopySummary:
    n_beats: int = 0
    n_ventricular: int = 0
    bigeminy: bool = False
    max_vt_run: int = 0  # longest run of consecutive ventricular beats
    vt: bool = False      # ventricular tachycardia (fast, sustained-enough run)

    @property
    def burden(self) -> float:
        return round(self.n_ventricular / self.n_beats, 3) if self.n_beats else 0.0


# VT requires a fast, several-beat run — tuned on MIT-BIH DS2 windows to a
# false-positive-free operating point (PPV 1.0) for a §10 critical alert:
# a shorter/slower run over-calls on misclassified-V clusters (PPV ~0.1).
_VT_MIN_RUN = 4
_VT_MAX_RR_MS = 600.0  # run rate > ~100 bpm


def summarize_ectopy(lead: np.ndarray, rpeaks, fs: float, model: BeatClassifier) -> EctopySummary:
    """Classify beats and summarise ventricular ectopy (count, burden, bigeminy, VT)."""
    X, idx = extract_beat_features(lead, rpeaks, fs)
    if len(X) == 0 or not model.fitted:
        return EctopySummary(n_beats=len(idx))
    labels = model.predict(X)
    is_v = [lab == "V" for lab in labels]
    n_v = sum(is_v)
    rp = np.asarray(rpeaks, dtype=int)
    # Longest run of consecutive ventricular beats, and whether any run qualifies
    # as VT (>=_VT_MIN_RUN beats at a tachycardic rate).
    max_run = 0
    vt = False
    i = 0
    while i < len(is_v):
        if is_v[i]:
            j = i
            while j < len(is_v) and is_v[j]:
                j += 1
            runlen = j - i
            max_run = max(max_run, runlen)
            if runlen >= _VT_MIN_RUN:
                rr = [(rp[idx[k + 1]] - rp[idx[k]]) / fs * 1000.0 for k in range(i, j - 1)]
                if rr and float(np.mean(rr)) <= _VT_MAX_RR_MS:
                    vt = True
            i = j
        else:
            i += 1
    # Bigeminy: a PVC every 2nd beat (gap 2 between ventricular beats). Trigeminy
    # was evaluated on MIT-BIH but is too fragile under imperfect V-detection to
    # report (0.0 sensitivity on DS2 windows).
    v_pos = [k for k, v in enumerate(is_v) if v]
    from itertools import pairwise
    gaps = [b - a for a, b in pairwise(v_pos)]
    bigeminy = False
    if len(gaps) >= 3 and max_run < 3:  # a run is VT, not an allorhythmia
        bigeminy = sum(g == 2 for g in gaps) / len(gaps) >= 0.6
    return EctopySummary(n_beats=len(is_v), n_ventricular=n_v, max_vt_run=max_run,
                         vt=vt, bigeminy=bigeminy)


# 12-lead PVC beat rule (D25). Chosen on development data only (Chapman/Ningbo +
# PTB-XL outside test fold 10): record-level sens 0.73/0.74, spec 0.965/0.968,
# versus 0.31/0.29 and 0.93/0.89 for the MIT-BIH LDA with >=2 V beats.
_PVC_PRE_MAX = 0.90       # premature: preceding RR <= 0.90 x median RR
_PVC_POST_MIN = 1.10      # compensatory pause: following RR >= 1.10 x median RR
_PVC_CORR_MAX = 0.90      # shape differs from the dominant beat (mean over leads)
_PVC_LEADS = ("II", "V1", "V5")
_PVC_HALF_S = 0.10


def detect_pvc_beats(sig, rpeaks, fs: float) -> list[int]:
    """Indices (into ``rpeaks``) of beats meeting the 12-lead PVC rule.

    A PVC is premature, followed by a compensatory pause, and shaped unlike
    the record's dominant beat in several leads at once (mean correlation to
    the per-lead median beat over II, V1 and V5). Interior beats only.
    """
    rp = np.asarray(rpeaks, dtype=int)
    if rp.size < 5:
        return []
    leads = [n for n in _PVC_LEADS if n in sig.lead_names] or list(sig.lead_names[:3])
    half = int(_PVC_HALF_S * fs)
    n_samples = sig.signal.shape[1]
    ok = [i for i, r in enumerate(rp) if r - half >= 0 and r + half < n_samples]
    if len(ok) < 5:
        return []
    segs = {n: np.stack([sig.lead(n)[rp[i] - half:rp[i] + half].astype(float) for i in ok])
            for n in leads}
    tmpl = {n: np.median(segs[n], axis=0) for n in leads}
    rr = np.diff(rp).astype(float)
    med = float(np.median(rr)) or 1.0
    out = []
    for k, i in enumerate(ok):
        if i == 0 or i >= len(rp) - 1:
            continue
        if rr[i - 1] / med > _PVC_PRE_MAX or rr[i] / med < _PVC_POST_MIN:
            continue
        corr = np.mean([np.corrcoef(segs[n][k], tmpl[n])[0, 1] for n in leads])
        if corr <= _PVC_CORR_MAX:
            out.append(i)
    return out


# Ventricular runs (non-sustained VT) on a 10-s strip (D40; first iteration D38).
# Chosen on MIT-BIH DS1+DS2, INCART windows and PTB-XL/Chapman development records.
_RUN_LEADS = ("II", "MLII", "V1", "V5")
_RUN_TEMPLATE_PRE_MIN = 0.92   # template beats: not premature (pre-RR >= 0.92 x reference RR)
_RUN_TEMPLATE_RR_PCT = 75.0    # reference RR: upper quartile of the strip RR (D94; the median is ectopic when ectopy dominates)
_RUN_CORR_MAX = 0.80           # every run beat: mean correlation to that template
_RUN_CORR_CORE = 0.70          # ... and at least one clearly abnormal beat
_RUN_MIN_BEATS = 3
_RUN_PRE_MAX = 0.90            # the run starts early: RR <= 0.90 x the RR before it
_RUN_PRE_INNER = 0.70          # ... or the run inside is clearly faster: mean inner RR <= 0.70 x the RR before it (D95)
_RUN_MAX_RR_MS = 600.0         # >= 100/min inside the run
# A beat recovered from the multi-lead envelope (D75) is marked at the envelope peak,
# not the rhythm lead's R: it is compared with the template at its best alignment
# within this shift. Off-R, a normal beat reads as abnormal and three in a row made a
# false VT run (PTB-XL 15108, 18619, 4885, D87). Detector beats keep the strict match:
# a shift for every beat lets wide complexes fit too (INCART VT 46 -> 42 of 86).
_RUN_RECOVERED_SHIFT_MS = 40.0
# A detector beat marked off its R by up to the same shift is rescued only when the
# shifted match is near-perfect: the same shape displaced, not a wide complex that
# partly lines up (a plain shift for every beat cost INCART VT 46 -> 42, D87).
_RUN_SHIFTED_RESCUE_MIN = 0.95


def _best_corr(x: np.ndarray, tmpl: np.ndarray, shift: int) -> float:
    """Max correlation of the template against ``x`` (template length + 2*shift)."""
    n = tmpl.size
    best = -1.0
    for s in range(2 * shift + 1):
        c = np.corrcoef(x[s:s + n], tmpl)[0, 1]
        if np.isfinite(c) and c > best:
            best = float(c)
    return best


def detect_ventricular_runs(sig, rpeaks, fs: float, recovered=None) -> list[tuple[int, int, float]]:
    """Runs of >= 3 consecutive premature, abnormally shaped beats at >= 100/min.

    Returns ``(first_beat, n_beats, rate_per_min)`` per run. The template is built
    from beats that are not premature, so a run that is a large share of the strip
    does not become the template itself; a run must be followed by a beat (a run
    cut by the strip end is indistinguishable from end-of-strip filter artefact).
    """
    rp = np.asarray(rpeaks, dtype=int)
    if rp.size < 6:
        return []
    leads = [n for n in _RUN_LEADS if n in sig.lead_names] or list(sig.lead_names[:3])
    half = int(_PVC_HALF_S * fs)
    n_samples = sig.signal.shape[1]
    ok = [i for i, r in enumerate(rp) if r - half >= 0 and r + half < n_samples]
    if len(ok) < 5:
        return []
    segs = {n: np.stack([sig.lead(n)[rp[i] - half:rp[i] + half].astype(float) for i in ok])
            for n in leads}
    rr = np.diff(rp).astype(float)
    med = float(np.percentile(rr, _RUN_TEMPLATE_RR_PCT)) or 1.0
    timing_normal = [k for k, i in enumerate(ok) if i >= 1 and rr[i - 1] >= _RUN_TEMPLATE_PRE_MIN * med]
    pick = timing_normal if len(timing_normal) >= 3 else list(range(len(ok)))
    tmpl = {n: np.median(segs[n][pick], axis=0) for n in leads}
    corr = np.ones(len(rp))
    shift = int(_RUN_RECOVERED_SHIFT_MS / 1000.0 * fs)
    rec = {int(r) for r in (recovered or [])}
    for k, i in enumerate(ok):
        corr[i] = np.mean([np.corrcoef(segs[n][k], tmpl[n])[0, 1] for n in leads])
        r = int(rp[i])
        if (corr[i] <= _RUN_CORR_MAX and shift and r - half - shift >= 0
                and r + half + shift < n_samples):
            shifted = float(np.mean([
                _best_corr(sig.lead(n)[r - half - shift:r + half + shift].astype(float), tmpl[n], shift)
                for n in leads]))
            if r in rec or shifted >= _RUN_SHIFTED_RESCUE_MIN:
                corr[i] = max(corr[i], shifted)
    abnormal = corr <= _RUN_CORR_MAX
    out = []
    j = 0
    while j < len(rp):
        if not abnormal[j]:
            j += 1
            continue
        e = j
        while e < len(rp) and abnormal[e]:
            e += 1
        inner_ms = rr[j:e - 1] / fs * 1000.0
        # Prematurity against the preceding cycle, not the strip median: with frequent
        # ectopy the median itself is short (D38 missed 19 INCART runs this way).
        if (e - j >= _RUN_MIN_BEATS and j >= 2 and e < len(rp)
                and float(corr[j:e].min()) <= _RUN_CORR_CORE
                and (rr[j - 1] <= _RUN_PRE_MAX * rr[j - 2]
                     or float(np.mean(rr[j:e - 1])) <= _RUN_PRE_INNER * rr[j - 2])
                and float(np.mean(inner_ms)) <= _RUN_MAX_RR_MS):
            out.append((j, e - j, round(60000.0 / float(np.mean(inner_ms)))))
        j = e
    return out


def vt_run_finding(runs: list[tuple[int, int, float]]):
    """ventricular_tachycardia (critical) for the longest detected run (D40)."""
    from cardiolens.contracts.findings import Evidence, Finding

    _, n, rate = max(runs, key=lambda r: r[1])
    return Finding(
        code="ventricular_tachycardia", group="rhythm",
        # The run detector sees premature beats shaped unlike the dominant QRS: VT and an
        # aberrated SVT run look alike to it (D92: PTB-XL 21033, 12490), and a run on a
        # 10 s strip is non-sustained by definition.
        label_ru=(f"Пробежка широких комплексов ({n} комплексов, ~{int(rate)}/мин): "
                  "вероятна неустойчивая ЖТ, аберрация не исключена"),
        label_uz=(f"Keng komplekslar yugurishi ({n} kompleks, ~{int(rate)}/daq): "
                  "ehtimoliy beqaror JT, aberratsiya istisno qilinmagan"),
        evidence=[Evidence(feature="run_beats", value=str(n)),
                  Evidence(feature="run_rate_per_min", value=str(int(rate))),
                  Evidence(feature="runs", value=str(len(runs)))],
        confidence=0.7, critical=True,
    )


def pvc_findings(pvc_idx: list[int], n_beats: int):
    """ventricular_bigeminy (PVC on alternate beats) or ventricular_ectopy (D25)."""
    from itertools import pairwise

    from cardiolens.contracts.findings import Evidence, Finding

    if not pvc_idx:
        return []
    burden = len(pvc_idx) / max(1, n_beats)
    ev = [Evidence(feature="pvc_count", value=str(len(pvc_idx))),
          Evidence(feature="beats", value=str(n_beats)),
          Evidence(feature="burden", value=f"{burden:.0%}")]
    gaps = [b - a for a, b in pairwise(pvc_idx)]
    if len(pvc_idx) >= 3 and sum(g == 2 for g in gaps) / len(gaps) >= 0.6:
        return [Finding(code="ventricular_bigeminy", group="rhythm",
                        label_ru=f"Желудочковая бигеминия (ЖЭ {burden:.0%} ударов)",
                        label_uz=f"Ventrikulyar bigeminiya (VES {burden:.0%})",
                        evidence=ev, confidence=0.7)]
    return [Finding(code="ventricular_ectopy", group="rhythm",
                    label_ru=f"Желудочковая экстрасистолия — {len(pvc_idx)} ЖЭ / {n_beats} уд.",
                    label_uz=f"Ventrikulyar ekstrasistoliya — {len(pvc_idx)} VES / {n_beats}",
                    evidence=ev, confidence=0.75)]
