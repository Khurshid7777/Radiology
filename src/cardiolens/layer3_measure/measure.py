"""Layer 3 — detection and measurements (TZ §6).

QRS detection and P/QRS/T delineation via neurokit2, interval measurement
(PR/QRS/QT + QTc Bazett/Fridericia/Framingham), heart rate / regularity / HRV,
frontal QRS axis, ST measurement at J+60/80 ms, Sokolow-Lyon voltage, and simple
morphology descriptors (peaked/flat/inverted T, U waves).

Where delineation is unreliable we record a note and leave the field ``None``
rather than fabricating a value (TZ §0).
"""

from __future__ import annotations

import warnings
from itertools import pairwise

import numpy as np

from cardiolens.contracts.features import (
    Axes,
    Intervals,
    Measurements,
    STMeasurement,
    TWave,
)
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer2_preprocess.quality import is_lead_off
from cardiolens.layer3_measure.atrial import (
    FLUTTER_BAND_BPM,
    FLUTTER_SPECTRAL_LEADS,
    P_TRAIN_LEADS,
    atrial_activity,
    fwave_index,
    p_train,
    p_train_dissociated,
)
from cardiolens.layer3_measure.delineation import (
    P_ONSET_MIN_FRACTION,
    P_ONSET_MIN_UV,
    envelope_floor,
    qrs_bounds_envelope,
    qt_multilead,
    refine_p_onsets,
    spatial_velocity,
    terminal_qrs_sign,
)
from cardiolens.layer3_measure.global_beat import (
    global_q_waves,
    global_right_precordial_st,
    global_terminal_s,
)
from cardiolens.layer3_measure.intervals import (
    qtc_bazett,
    qtc_framingham,
    qtc_fridericia,
)
from cardiolens.layer3_measure.voltages import measure_voltages

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Lead preferred for rhythm/delineation.
_RHYTHM_LEAD_PREFERENCE = ["II", "V5", "V2", "I"]

# A beat's P wave counts as "present/organized" when its P-region peak-to-peak is at
# least this fraction of the R-wave amplitude reference. Below it (e.g. AF fibrillatory
# waves) the beat contributes to a low p_wave_ratio, which the AF rule keys on (§7.1).
# Calibrated on the PhysioNet ecg-arrhythmia set (sinus P/R ~0.21 vs AF ~0.10).
P_PRESENCE_FRACTION = 0.15

# PR-segment window used as the ST baseline: [onset-50 ms, onset-10 ms].
ST_BASELINE_MS = (50.0, 10.0)
J_REFINE_MS = 40                 # per-lead J may move this far past the envelope QRS offset
J_REFINE_SLOPE_FRACTION = 0.15   # ... to where the lead's slope is this fraction of its QRS slope

# Non-QRS detection filter: envelope peak below this fraction of the 90th-percentile
# envelope peak across detections = P or T wave, not a beat.
OVERSENSE_ENV_FRACTION = 0.25

# Beat-to-beat PR standard deviation below this = a fixed PR = conducted rhythm;
# an AV-dissociation candidate with a stable PR is rejected.
PR_STD_DISSOCIATION_MS = 40.0
# D112 (plan step 5a): AV dissociation from the QRS-independent P train instead of the
# residual autocorrelation and its PR vetoes. Experiment until the held-out check.
ATRIAL_P_TRAIN = False
# ... and a dissociation candidate needs organised P waves (AF has none).
AV_DISSOCIATION_MIN_P_RATIO = 0.3


def _lacks_organised_p(p_wave_ratio: float | None, n_beats: int) -> bool:
    """Too few beats with an organised P for AV dissociation (AF with a slow response).

    A complete block at 35-45/min leaves only 6-7 beats on a 10 s strip, so one beat
    more or less at the record edge moved the plain share across the threshold (2/6 =
    0.33 passed, 2/7 = 0.29 vetoed a synthetic block). The share is taken with one beat
    of slack, (P + 1) / (beats + 1): a paced record with P before 2 of 11 beats (PTB-XL
    12000) is still vetoed, and on PTB-XL no other record reached this veto.
    """
    n_p = round((p_wave_ratio or 0.0) * n_beats)
    return (n_p + 1) / (n_beats + 1) < AV_DISSOCIATION_MIN_P_RATIO


def _pick_rhythm_lead(sig: SignalRepresentation) -> int:
    for name in _RHYTHM_LEAD_PREFERENCE:
        if name in sig.lead_names:
            return sig.lead_names.index(name)
    return 0


def _reject_non_qrs_detections(
    signal: np.ndarray, rpeaks: np.ndarray, fs: float,
    windows: dict[str, tuple[float, float]] | None = None, lead_names: list[str] | None = None,
) -> np.ndarray:
    """Drop peak detections that are a P or T wave, not a QRS.

    At slow rates or after long pauses the adaptive peak detector accepts a tall
    T wave or a P wave as a beat. Amplitude cannot tell them apart robustly (a
    small conducted beat next to a tall PVC, a lead-II R flattened by axis
    deviation), but slope can: the multi-lead spatial-velocity envelope peaks at
    >= 0.7 of the record's reference on every true QRS and at <= 0.11 on P/T waves
    (PTB-XL and synthetic checks). A detection below OVERSENSE_ENV_FRACTION of the
    90th-percentile envelope peak is discarded.

    On a digitised 3x4 page each column carries a different set of leads (the
    other channels are zero there), so the envelope of a beat under aVR/aVL/aVF
    is a fraction of one under V1–V3 and a whole column of true beats would be
    rejected against a reference set by the chest leads (D52). With lead windows
    given, beats are grouped by the leads valid at their time, the envelope is
    built from those leads only and the reference is taken within the group; a
    group too small for a percentile (< 3 beats) keeps all its beats.
    """
    if len(rpeaks) < 3:
        return rpeaks
    w = int(0.03 * fs)
    names = lead_names or []
    valid_sets: list[tuple[int, ...]] = []
    for r in rpeaks:
        t = float(r) / fs
        if windows and names:
            rows = tuple(i for i, ln in enumerate(names) if ln not in windows or windows[ln][0] <= t <= windows[ln][1])
        else:
            rows = tuple(range(signal.shape[0]))
        valid_sets.append(rows)
    keep = np.ones(len(rpeaks), dtype=bool)
    for rows in set(valid_sets):
        members = np.flatnonzero([vs == rows for vs in valid_sets])
        if len(members) < 3 or not rows:
            continue
        env = spatial_velocity(signal[list(rows)], fs)
        peaks = np.array([float(env[max(0, int(rpeaks[i]) - w):int(rpeaks[i]) + w].max()) for i in members])
        ref = float(np.percentile(peaks, 90))
        if ref <= 0:
            continue
        keep[members] = peaks >= OVERSENSE_ENV_FRACTION * ref
    return rpeaks[keep]



RECOVER_ENV_FRACTION = 0.5    # a recovered QRS reaches half the accepted beats' median envelope peak
RECOVER_MIN_GAP_S = 0.25      # ... and lies a refractory period from every accepted beat: a wide
                              # (LBBB) QRS has two envelope peaks ~130 ms apart (PTB-XL 13820)
RECOVER_REF_HALF_S = 0.15     # an accepted beat's envelope level = max within this of its R (QRS search half-width)
RECOVER_MAX_FLOOR = 0.25      # no recovery when the envelope between beats is this high relative
                              # to the beats: noise, not QRS, reaches half their level (PTB-XL 15973)


PACED_R_MAX_OFFSET_S = 0.010   # a detection this close to a pacing spike sits on the spike
PACED_QRS_WINDOW_S = (0.08, 0.30)   # an atrially paced QRS follows its spike after the AV delay
PACED_QUIET_S = (0.02, 0.07)   # ... with a quiet envelope right after the spike (a ventricular
PACED_QUIET_FRACTION = 0.3     #     spike is followed at once by its own wide QRS: left alone)
PACED_QRS_MIN_RATIO = 2.3      # the QRS outweighs the spike point on the envelope (atrial pacing
                               # 2.5-4.2; a late hump of a ventricular paced beat 1.1-2.0)


def _move_off_pacing_spikes(signal: np.ndarray, rpeaks: np.ndarray, fs: float,
                            spikes_s: list[float] | None) -> tuple[np.ndarray, int]:
    """Move detections that sit on an atrial pacing spike onto the QRS that follows (D98).

    With atrial pacing the single-lead detector can lock onto the spikes instead of the
    QRS 250-300 ms later; the oversense gate then keeps some of them and HR halves
    (Georgia E00850: 33/min at 60). A detection within PACED_R_MAX_OFFSET_S of a raw
    pacing spike, with a quiet envelope just after the spike and an envelope peak in
    PACED_QRS_WINDOW_S, is moved to that peak; one that lands on an already detected
    QRS is dropped. Ventricular pacing (QRS right after the spike) is left alone.
    """
    if not spikes_s or len(rpeaks) == 0:
        return rpeaks, 0
    env = spatial_velocity(signal, fs)
    sp = np.asarray(spikes_s, dtype=float) * fs
    q0, q1 = (int(x * fs) for x in PACED_QUIET_S)
    w0, w1 = (int(x * fs) for x in PACED_QRS_WINDOW_S)
    out, moved = [], 0
    for r in (int(x) for x in rpeaks):
        k = int(np.argmin(np.abs(sp - r)))
        s0 = int(round(sp[k]))
        if abs(sp[k] - r) > PACED_R_MAX_OFFSET_S * fs or s0 + w1 >= env.size:
            out.append(r)
            continue
        win = env[s0 + w0:s0 + w1]
        peak = s0 + w0 + int(np.argmax(win))
        at_r = float(env[max(0, r - q0):r + q0].max())
        if (float(env[s0 + q0:s0 + q1].max()) >= PACED_QUIET_FRACTION * float(env[peak])
                or float(env[peak]) < PACED_QRS_MIN_RATIO * at_r):
            out.append(r)
            continue
        out.append(peak)
        moved += 1
    if not moved:
        return rpeaks, 0
    merged = []
    for r in sorted(out):
        if merged and r - merged[-1] < int(0.25 * fs):
            continue
        merged.append(r)
    return np.asarray(merged, dtype=rpeaks.dtype), moved


def _recover_missed_qrs(signal: np.ndarray, rpeaks: np.ndarray, fs: float,
                        notes: list[str] | None = None) -> tuple[np.ndarray, int]:
    """Add QRS complexes the single-lead detector missed (D75).

    The primary detector runs on one lead. Where the QRS is small in that lead and the
    T or P wave tall (low-voltage II, large T), it locks onto T/P; the oversense gate
    then rightly drops those and the true complexes were never detected: a 104/min
    rhythm read as 22/min (PTB-XL 18191) and raised a false severe-bradycardia alert.
    The multi-lead spatial-velocity envelope peaks on every QRS, so peaks of it at >=
    RECOVER_ENV_FRACTION of the accepted beats' median (P/T reach <= 0.11 of the
    reference, D29), at least a refractory 250 ms apart and RECOVER_MIN_GAP_S from any
    accepted beat, are added — only when the rhythm becomes more regular with them.
    """
    if len(rpeaks) < 2:
        return rpeaks, 0
    from scipy.signal import find_peaks

    env = spatial_velocity(signal, fs)
    # The reference is each accepted beat's whole complex, not the detected point: on a
    # wide (LBBB) QRS the detector often sits on the weaker first of two envelope humps,
    # and half of that let the T waves through as beats (13820: 124/min at 75).
    w = int(RECOVER_REF_HALF_S * fs)
    ref = float(np.median([env[max(0, int(r) - w):int(r) + w].max() for r in rpeaks]))
    if ref <= 0:
        return rpeaks, 0
    # Quality gate: on a noisy strip the accepted "beats" are themselves partly noise and
    # noise peaks pass the half-level test (15973, floor 0.45 of the beats: 33 -> 164/min,
    # true 44). Where recovery helped on PTB-XL the floor was <= 0.15 (median 0.03).
    floor = envelope_floor(env, rpeaks)
    if floor > RECOVER_MAX_FLOOR * ref:
        if notes is not None:
            notes.append(f"QRS recovery skipped: noisy envelope (floor {floor / ref:.2f} of beats) (D79)")
        return rpeaks, 0
    cand, _ = find_peaks(env, height=RECOVER_ENV_FRACTION * ref, distance=max(1, int(0.25 * fs)))
    gap = RECOVER_MIN_GAP_S * fs
    new = [int(c) for c in cand if np.min(np.abs(rpeaks - c)) > gap]
    if not new:
        return rpeaks, 0
    merged = np.unique(np.concatenate([rpeaks, np.asarray(new, dtype=rpeaks.dtype)]))
    # Missed complexes fill the long gaps and make the rhythm more regular; pacemaker
    # spikes or notched/wide complexes add peaks inside normal RR intervals and make it
    # less regular (paced PTB-XL records read 70 -> 120-140/min without this check).
    def _cv(b):
        rr = np.diff(b.astype(float))
        return float(np.std(rr) / (np.mean(rr) + 1e-9)) if rr.size >= 2 else float("inf")
    if _cv(merged) >= _cv(rpeaks):
        return rpeaks, 0
    return merged, len(new)

DROPPED_BEAT_RR_RANGE = (1.8, 2.2)   # an RR this many medians long is one missed detection
DROPPED_BEAT_MAX_FRACTION = 0.25     # more such RRs than this is a rhythm, not detector loss


def _rr_for_regularity(rr: np.ndarray) -> tuple[np.ndarray, int]:
    """RR series for the regularity statistic with isolated missed beats healed.

    One undetected R (a small complex on a digitised page, a beat under
    artefact) leaves a single RR of twice the median; on a 10-s strip that one
    interval alone drives the RR coefficient of variation past the AF gate. An
    RR within ``DROPPED_BEAT_RR_RANGE`` of the median is split into two halves
    for the statistic, as long as such intervals are a minority
    (``DROPPED_BEAT_MAX_FRACTION``): in 2:1 block the long interval *is* the
    median and nothing is touched; in AF the intervals do not cluster at 2x.
    HR and the stored R peaks are left as measured. Returns the series and the
    number of healed intervals.
    """
    if len(rr) < 3:
        return rr, 0
    med = float(np.median(rr))
    lo, hi = DROPPED_BEAT_RR_RANGE
    long = (rr >= lo * med) & (rr <= hi * med)
    n = int(long.sum())
    if n == 0 or n > DROPPED_BEAT_MAX_FRACTION * len(rr):
        return rr, 0
    out = []
    for v, is_long in zip(rr, long):
        out.extend([v / 2.0, v / 2.0] if is_long else [v])
    return np.asarray(out), n


def _lead_off(sig: SignalRepresentation, lead: str) -> bool:
    """True when ``lead`` carries no signal (Layer-2 lead-off criterion)."""
    return lead in sig.lead_names and is_lead_off(sig.lead(lead))


def _lead_mask(sig: SignalRepresentation, lead: str, rpeaks: np.ndarray, fs: float) -> np.ndarray:
    """Beats that fall inside ``lead``'s valid time window (all, when no windows).

    A digitised 3x4 page carries each lead for 2.5 s only; per-lead statistics
    must ignore beats outside that window (the samples there are zero, not data).
    A lead that carries no signal at all (lead-off / unrecorded channel) has no
    valid beats — callers that fall back to "all beats" on an empty mask must
    check ``_lead_off`` first (D45).
    """
    if _lead_off(sig, lead):
        return np.zeros(len(rpeaks), dtype=bool)
    windows = getattr(sig.provenance, "lead_windows", None)
    if not windows or lead not in windows:
        return np.ones(len(rpeaks), dtype=bool)
    t0, t1 = windows[lead]
    t = np.asarray(rpeaks, dtype=float) / fs
    pad = 0.15  # keep whole complexes away from the window edges
    return (t >= t0 + pad) & (t <= t1 - pad)


def _aligned(values, rpeaks: np.ndarray, before: bool) -> np.ndarray:
    """neurokit wave marks as a float array, one slot per R peak (NaN where missing).

    neurokit can return fewer marks than beats (it drops beats at the record edge);
    padding by order then shifts every later mark to the wrong beat (D90: 14323,
    PR -1031 ms). On a count mismatch each mark goes to its own beat by position:
    an onset (``before``) to the next R, an offset to the previous R.
    """
    r = np.asarray(rpeaks)
    n = len(r)
    arr = np.full(n, np.nan)
    vals = np.asarray(list(values), dtype=float)
    if len(vals) == n:
        return vals
    for v in vals[np.isfinite(vals)]:
        k = int(np.searchsorted(r, v, side="left")) if before else int(np.searchsorted(r, v, side="right")) - 1
        if 0 <= k < n and not np.isfinite(arr[k]):
            arr[k] = v
    return arr


def _median_ms(values: list[float], fs: float) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.median(vals) / fs * 1000.0)


def _delineate(cleaned: np.ndarray, rpeaks: np.ndarray, fs: float):
    """Return per-beat onset/offset arrays via neurokit2 (best-effort)."""
    import neurokit2 as nk

    try:
        _, waves = nk.ecg_delineate(
            cleaned, {"ECG_R_Peaks": rpeaks}, sampling_rate=fs, method="dwt"
        )
        return waves
    except Exception:
        return None


PAGE_BEAT_MIN_FRACTION = 0.35   # envelope peak, as a fraction of its column's 90th-percentile peak
PAGE_BEAT_MIN_RR_S = 0.24       # refractory distance between beats (250 bpm)
PAGE_BEAT_SNAP_S = 0.06         # a beat is snapped to the column's summed-deflection apex within this window
PAGE_BEAT_EDGE_START_S = 0.30   # no beat without a full P-QRS window before it
PAGE_BEAT_EDGE_END_S = 0.45     # no beat without a T wave window after it
PAGE_BEAT_RHYTHM_FRACTION = 0.3 # the rhythm lead's own extreme is used when its deflection is this fraction of the column's
PAGE_BEAT_MATCH_S = 0.10        # a single-lead beat within this of a column beat is the same beat
PAGE_BEAT_ADD_FRACTION = 0.6    # a column beat with no single-lead partner is added only above this envelope height
PAGE_BEAT_MIN_AGREEMENT = 0.6   # the columns are trusted when they confirm this share of the judgeable single-lead beats, or when their list is the more regular one
PAGE_BEAT_ADD_MIN_RR = 0.6      # an added beat needs this fraction of the confirmed median RR to each neighbour (room for a missed beat)
PAGE_BEAT_DROP_MAX_GAP = 1.6    # dropping a beat may not open a gap longer than this × the confirmed median RR (a true beat the columns missed)
PAGE_BEAT_GATE_MIN_CONFIRMED = 5  # fewer confirmed beats define no rhythm (three gave a median RR twice the true one)
# T-wave measurement validity (D61): a deflection narrower than this at half amplitude
# is a QRS remnant, not a T wave; the T window ends this long before the next beat.
T_MIN_HALF_WIDTH_S = 0.04
T_NEXT_BEAT_GUARD_S = 0.20


def _fit_page_edits_to_rhythm(confirmed: list, kept: list, added: list, rpeaks: np.ndarray) -> tuple[list, list]:
    """Keep only the column-detector edits that fit the rhythm the confirmed beats
    define (D55). An added beat needs room for itself — PAGE_BEAT_ADD_MIN_RR of
    the confirmed median RR to each kept neighbour; dropping a single-lead beat may
    not open a gap longer than PAGE_BEAT_DROP_MAX_GAP median RRs *after the
    additions* (a gap the columns fill with their own beat is no gap; a gap nobody
    fills is a true beat the columns missed). Fewer than
    PAGE_BEAT_GATE_MIN_CONFIRMED confirmed beats define no rhythm — on a page whose
    single-lead list sat on the P waves, three confirmed beats gave a median RR
    twice the true one and the gate rejected the true beats — and the edits pass
    unchanged, left to the agreement / regularity decision."""
    if len(confirmed) < PAGE_BEAT_GATE_MIN_CONFIRMED:
        return kept, added
    med_rr = float(np.median(np.diff(np.sort(np.asarray(confirmed, dtype=float)))))
    base = np.sort(np.asarray(kept, dtype=int))

    def _neighbours(x, arr):
        left, right = arr[arr < x], arr[arr > x]
        return (left[-1] if left.size else None), (right[0] if right.size else None)

    def _has_room(p):
        left, right = _neighbours(p, base)
        return ((left is None or p - left >= PAGE_BEAT_ADD_MIN_RR * med_rr)
                and (right is None or right - p >= PAGE_BEAT_ADD_MIN_RR * med_rr))

    added = [p for p in added if _has_room(p)]
    kept = list(kept)
    merged = np.sort(np.asarray(kept + added, dtype=int))
    for r in sorted(set(np.asarray(rpeaks).tolist()) - set(kept)):
        left, right = _neighbours(r, merged)
        if left is not None and right is not None and right - left > PAGE_BEAT_DROP_MAX_GAP * med_rr:
            kept.append(r)
            merged = np.sort(np.append(merged, r))
    return kept, added


def _page_beats(sig: SignalRepresentation, windows: dict, rhythm_row: np.ndarray,
                fs: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Beat times on a 3x4 page from every lead valid at each moment (D55).

    The rhythm strip alone drops beats when lead II is near-isoelectric (a 0.2-mV
    QRS with a 20-ms S lost to the page) or a P wave outranks a small R under
    LBBB. Each time column of the page carries four leads; the spatial-velocity
    envelope over the leads valid in that column peaks on every true QRS. The
    envelope is normalised per column (its own 90th-percentile peak, so a column
    of limb leads is not judged against the chest leads), beats are its peaks
    above ``PAGE_BEAT_MIN_FRACTION`` with a ``PAGE_BEAT_MIN_RR_S`` refractory
    distance, and each beat is snapped to the rhythm lead's nearest extreme so
    downstream delineation sees an R on the strip. Returns None when the page
    has no column structure.
    """
    from scipy.signal import find_peaks

    n = sig.signal.shape[1]
    t = np.arange(n) / fs
    valid = np.zeros((len(sig.lead_names), n), dtype=bool)
    for i, ln in enumerate(sig.lead_names):
        if ln in windows:
            t0, t1 = windows[ln]
            valid[i] = (t >= t0) & (t <= t1)
        else:
            valid[i] = True
    starts = sorted({0} | {int(round(w[0] * fs)) for w in windows.values()} | {n})
    cols = [(a, b) for a, b in pairwise(starts) if b - a > int(0.5 * fs)]
    # One column = a strip where every lead runs over the whole record (12x1
    # layout, D72): the envelope over all leads is the point, so it still applies
    # — but only when the windows say so; no windows at all means no page.
    if len(cols) < 1 or (len(cols) < 2 and not windows):
        return None
    env = np.zeros(n)
    for a, b in cols:
        rows = [i for i in range(len(sig.lead_names)) if valid[i, a:b].all() and not is_lead_off(sig.signal[i, a:b])]
        if not rows:
            continue
        e = spatial_velocity(sig.signal[rows, a:b], fs)
        pk, _ = find_peaks(e, distance=max(1, int(PAGE_BEAT_MIN_RR_S * fs)))
        ref = float(np.percentile(e[pk], 90)) if pk.size >= 3 else float(e.max())
        if ref > 0:
            env[a:b] = e / ref
    peaks, props = find_peaks(env, height=PAGE_BEAT_MIN_FRACTION, distance=max(1, int(PAGE_BEAT_MIN_RR_S * fs)))
    heights = props["peak_heights"]
    # The page's first/last columns start on a filter transient, and a beat needs
    # a full P-QRS before it and a T wave after it for the delineation that
    # follows (the single-lead detector skips such edge beats too).
    inside = (peaks >= int(PAGE_BEAT_EDGE_START_S * fs)) & (peaks < n - int(PAGE_BEAT_EDGE_END_S * fs))
    peaks, heights = peaks[inside], heights[inside]
    if peaks.size < 2:
        return None
    # The envelope peaks on the steepest slope, not the apex: snap each beat to
    # the moment of maximal summed |deflection| over the column's valid leads
    # (the R/S apex seen by all of them), so the beat times are as steady as on
    # the digital record and a near-isoelectric rhythm lead cannot jitter them.
    w = int(PAGE_BEAT_SNAP_S * fs)
    col_rows = {}
    for a, b in cols:
        rows = [i for i in range(len(sig.lead_names)) if valid[i, a:b].all() and not is_lead_off(sig.signal[i, a:b])]
        col_rows[(a, b)] = (rows, np.median(sig.signal[rows, a:b], axis=1) if rows else None)
    # Downstream delineation is anchored on the rhythm lead's R, so a beat is
    # placed on that lead's own extreme whenever it carries a clear complex
    # there (its deflection at least PAGE_BEAT_RHYTHM_FRACTION of the column's
    # mean per-lead deflection); only a near-flat rhythm lead falls back to the
    # column apex — otherwise the QRS/QT search windows would shift.
    rhythm_base = float(np.median(rhythm_row))
    snapped, kept_heights = [], []
    for p, hgt in zip(peaks, heights, strict=True):
        kept_heights.append(float(hgt))
        key = next(((a, b) for a, b in cols if a <= p < b), None)
        rows, base = col_rows.get(key, ([], None))
        lo, hi = max(0, p - w), min(n, p + w + 1)
        own_pos = rhythm_row[lo:hi] - rhythm_base            # the R the single-lead detector marks
        own = np.abs(own_pos)
        if rows:
            per_lead = np.abs(sig.signal[rows, lo:hi] - base[:, None])
            level = PAGE_BEAT_RHYTHM_FRACTION * per_lead.max(axis=1).mean()
            if own_pos.max() >= level:
                seg = own_pos
            elif own.max() >= level:
                seg = own
            else:
                seg = per_lead.sum(axis=0)
        else:
            seg = own_pos if own_pos.max() > 0 else own
        snapped.append(lo + int(np.argmax(seg)))
    pos = np.asarray(snapped, dtype=int)
    order = np.argsort(pos)
    return pos[order], np.asarray(kept_heights)[order]


def measure(sig: SignalRepresentation, pacing_spikes_s: list[float] | None = None) -> Measurements:
    """Full Layer-3 measurement pass (TZ §6). ``pacing_spikes_s``: raw-signal pacing
    spike times (the band-pass erases them), used to keep R off atrial pacing spikes."""
    import neurokit2 as nk

    fs = sig.sampling_rate_hz
    m = Measurements()
    lead_idx = _pick_rhythm_lead(sig)
    cleaned = sig.signal[lead_idx].astype(np.float64)

    # --- R-peaks, HR, regularity, HRV ---
    try:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        rpeaks = np.array([], dtype=int)
    rpeaks, n_moved = _move_off_pacing_spikes(sig.signal, rpeaks, fs, pacing_spikes_s)
    m.atrial_paced_beats = n_moved
    if n_moved:
        m.notes.append(f"{n_moved} R detection(s) moved off atrial pacing spikes onto the QRS (D98)")
    page_windows = getattr(sig.provenance, "lead_windows", None)
    page_confirmed = False
    if page_windows:
        # The single-lead detector both drops beats (small complexes lost to the
        # page) and adds P/T waves on the strip; the column detector sees four
        # leads at every moment and replaces it whenever it finds a rhythm.
        found = _page_beats(sig, page_windows, cleaned, fs)
        if found is not None and len(found[0]) >= 3:
            page, heights = found
            # Merge, do not replace: a single-lead beat confirmed by the column
            # detector keeps its own position (the R the delineation is tuned
            # to); an unconfirmed one is a P/T oversense and goes; a column beat
            # with no single-lead partner is a missed beat and is added when its
            # envelope is unmistakably a QRS (a tall T wave of a slow rhythm
            # reaches the confirmation level but not this one).
            tol = int(PAGE_BEAT_MATCH_S * fs)
            # A beat on a column boundary or at the page's time edges is clipped
            # in the panel leads and cannot be judged by the columns: keep it.
            n_s = sig.signal.shape[1]
            edges = np.array(sorted({int(round(w[0] * fs)) for w in page_windows.values()} | {0, n_s}))
            def _unjudgeable(r):
                return (np.min(np.abs(edges - r)) <= tol or r < int(PAGE_BEAT_EDGE_START_S * fs)
                        or r >= n_s - int(PAGE_BEAT_EDGE_END_S * fs))
            confirmed = [r for r in rpeaks if page.size and np.min(np.abs(page - r)) <= tol]
            # Such a beat is judged on the rhythm strip alone, which runs across
            # every boundary: its slope must reach half of the confirmed beats'.
            slope = spatial_velocity(cleaned[None, :], fs)
            w_s = int(0.03 * fs)
            s_at = lambda r: float(slope[max(0, int(r) - w_s):int(r) + w_s].max())
            s_ref = float(np.median([s_at(r) for r in confirmed])) if confirmed else 0.0
            kept = [r for r in rpeaks if (r in confirmed) or (_unjudgeable(r) and s_at(r) >= 0.5 * s_ref)]
            added = [p for p, hgt in zip(page, heights, strict=True)
                     if hgt >= PAGE_BEAT_ADD_FRACTION and (not rpeaks.size or np.min(np.abs(rpeaks - p)) > tol)]
            # Both edits must fit the rhythm the confirmed beats define. On a
            # badly extracted photo the column envelopes peak on noise and the
            # merge added a third of the beats inside normal RR intervals (HR
            # 69 -> 95 on a page read at r 0.77) or dropped true beats the
            # columns failed to confirm (HR 100 -> 79). A missed beat has room
            # for itself; dropping a true beat opens a gap no rhythm has.
            kept, added = _fit_page_edits_to_rhythm(confirmed, kept, added, rpeaks)
            # On a badly extracted page (a strong photo) the column envelopes are
            # noise and would re-write a sound single-lead list: the columns are
            # trusted only when they confirm most of the judgeable single-lead beats.
            judged = [r for r in rpeaks if not _unjudgeable(r)]
            agreement = len([r for r in judged if r in confirmed]) / max(1, len(judged))
            merged = np.unique(np.asarray(kept + added, dtype=int))

            def _cv(beats):
                rr = np.diff(np.sort(np.asarray(beats, dtype=float)))
                return float(np.std(rr) / (np.mean(rr) + 1e-9)) if len(rr) >= 2 else float("inf")
            # When the two detectors disagree, the more regular list is the rhythm:
            # P/T oversenses on the strip and noise peaks on a bad page both make
            # the offending list more irregular than the true beats.
            columns_trusted = agreement >= PAGE_BEAT_MIN_AGREEMENT or _cv(merged) < _cv(rpeaks)
            if len(merged) >= 3 and columns_trusted:
                n_add, n_drop = len(added), len(rpeaks) - len(kept)
                rpeaks = merged
                page_confirmed = True
                if n_add or n_drop:
                    m.notes.append(f"page beats: {n_add} added, {n_drop} dropped by the column detector (D55)")

    # The column detector has already judged every page beat within its own
    # column; the single-lead oversense gate would re-judge a confirmed beat at
    # the rhythm lead's R, up to 50 ms from the envelope peak, and drop it.
    if not page_confirmed:
        rpeaks = _reject_non_qrs_detections(
            sig.signal, rpeaks, fs, windows=getattr(sig.provenance, "lead_windows", None), lead_names=sig.lead_names)
        if not getattr(sig.provenance, "lead_windows", None):
            before = {int(r) for r in rpeaks}
            rpeaks, n_rec = _recover_missed_qrs(sig.signal, rpeaks, fs, m.notes)
            if n_rec:
                m.notes.append(f"{n_rec} QRS recovered from the multi-lead envelope (D75)")
                m.r_peaks_recovered = sorted(int(r) for r in rpeaks if int(r) not in before)
    m.r_peaks_samples = rpeaks.tolist()
    m.qrs_template_consistency = _qrs_template_consistency(sig, rpeaks, fs)
    m.qrs_partner_consistency = _qrs_partner_consistency(sig, rpeaks, fs)
    # Median-beat Q waves and right-precordial ST (ported from claude/vibrant-curie-ro1zsz
    # D36, D89, D90): pathological Q and the Brugada type-1 pattern.
    if len(rpeaks) >= 3:
        names = list(sig.lead_names)
        m.morphology.q_waves = global_q_waves(sig.signal, rpeaks, fs, names)
        m.morphology.right_precordial_st = global_right_precordial_st(sig.signal, rpeaks, fs, names)
        m.morphology.terminal_s_lateral = global_terminal_s(sig.signal, rpeaks, fs, names)
    m.n_beats = len(rpeaks)

    rr_s_mean = None
    if len(rpeaks) >= 2:
        rr = np.diff(rpeaks) / fs  # seconds
        rr_s_mean = float(np.mean(rr))
        m.heart_rate_bpm = round(60.0 / rr_s_mean, 1)
        m.rr_std_ms = round(float(np.std(rr) * 1000.0), 1)
        m.hrv_sdnn_ms = round(float(np.std(rr) * 1000.0), 1)
        # Regularity: coefficient of variation of RR < 10% -> regular. An isolated
        # RR of twice the median is a dropped detection, not irregularity (D52).
        rr_reg, dropped = _rr_for_regularity(rr)
        cv = float(np.std(rr_reg) / (np.mean(rr_reg) + 1e-9))
        m.rr_cv = round(cv, 3)
        m.rhythm_regular = bool(cv < 0.10)
        med = float(np.median(rr))
        m.rr_mad_cv = round(1.4826 * float(np.median(np.abs(rr - med))) / (med + 1e-9), 3)
        m.max_rr_ms = round(float(np.max(rr)) * 1000.0, 1)
        m.pause_ratio = round(float(np.max(rr) / (med + 1e-9)), 2)
        if dropped:
            m.notes.append(f"{dropped} RR interval(s) of twice the median: possible missed beat(s), "
                           "excluded from the regularity statistic")
    else:
        m.notes.append("insufficient R-peaks for rhythm analysis (low confidence)")

    # --- QRS bounds from the multi-lead envelope + dwt for P/T ---
    # The spatial-velocity envelope gives QRS onsets/offsets that hold up on real
    # BBB morphology (dwt under-estimates width); dwt still supplies P and T marks.
    n_beats = len(rpeaks)
    qrs_on = np.full(n_beats, np.nan)
    qrs_off = np.full(n_beats, np.nan)
    t_off: np.ndarray | None = None
    p_on = p_off = None
    if n_beats >= 3:
        qrs_on, qrs_off = qrs_bounds_envelope(sig.signal, rpeaks, fs)
        p_on = p_off = None
        waves = _delineate(cleaned, rpeaks, fs)
        if waves is not None:
            p_on = _aligned(waves.get("ECG_P_Onsets", []), rpeaks, before=True)
            p_off = _aligned(waves.get("ECG_P_Offsets", []), rpeaks, before=True)
            dwt_on = _aligned(waves.get("ECG_R_Onsets", []), rpeaks, before=True)
            dwt_off = _aligned(waves.get("ECG_R_Offsets", []), rpeaks, before=False)
            t_off = _aligned(waves.get("ECG_T_Offsets", []), rpeaks, before=False)
            # Fall back to dwt bounds only where the envelope found nothing.
            qrs_on = np.where(np.isfinite(qrs_on), qrs_on, dwt_on)
            qrs_off = np.where(np.isfinite(qrs_off), qrs_off, dwt_off)
        else:
            m.notes.append("delineation failed (P/T marks unavailable, low confidence)")

        base = float(np.median(cleaned))
        r_ref = float(np.percentile(np.abs(cleaned - base), 99)) or 1.0

        # PR from amplitude-refined P onsets: dwt onsets land late on real signals,
        # which under-estimates PR and hides first-degree AV block.
        p_on_ref = refine_p_onsets(
            cleaned, qrs_on, t_off, fs, max(P_ONSET_MIN_UV, P_ONSET_MIN_FRACTION * r_ref))
        pr_list = [qrs_on[i] - p_on_ref[i] for i in range(n_beats)
                   if np.isfinite(p_on_ref[i]) and np.isfinite(qrs_on[i])]
        m.p_onset_ratio = round(len(pr_list) / max(1, n_beats), 2)
        if len(pr_list) >= 3:
            m.pr_std_ms = round(float(np.std(pr_list) / fs * 1000.0), 1)
        if len(pr_list) < 0.5 * n_beats:
            pr_list = ([qrs_on[i] - p_on[i] for i in range(n_beats)
                        if np.isfinite(p_on[i]) and np.isfinite(qrs_on[i])]
                       if p_on is not None else [])
            if pr_list:
                m.notes.append("PR from dwt P onsets (refined P on <50% of beats)")
        qrs_list = [qrs_off[i] - qrs_on[i] for i in range(n_beats)
                    if np.isfinite(qrs_on[i]) and np.isfinite(qrs_off[i])
                    and qrs_off[i] > qrs_on[i]]
        # QT: tangent-method T offset per lead, measured in the lead with the longest
        # QT (guideline practice); the single-lead dwt offset is only a fallback.
        qt_tan, qt_lead, t_off_tan = qt_multilead(
            sig.signal, sig.lead_names, rpeaks, qrs_on, qrs_off, fs,
            masks={ln: _lead_mask(sig, ln, rpeaks, fs) for ln in sig.lead_names})
        if qt_tan is not None:
            qt_ms, qt_source = qt_tan, "tangent"
            t_off = t_off_tan
        else:
            qt_list = ([t_off[i] - qrs_on[i] for i in range(n_beats)
                        if np.isfinite(t_off[i]) and np.isfinite(qrs_on[i])]
                       if t_off is not None else [])
            qt_ms, qt_source = _median_ms(qt_list, fs), ("dwt" if qt_list else None)
            if qt_ms is not None:
                m.notes.append("QT from dwt T offset (tangent method found no usable lead)")
        m.intervals = Intervals(
            pr_ms=_median_ms(pr_list, fs),
            qrs_ms=_median_ms(qrs_list, fs),
            qt_ms=qt_ms, qt_lead=qt_lead, qt_source=qt_source,
        )

        if p_on is not None and p_off is not None:
            # Fraction of beats with an *organized* P wave, by amplitude (low ->
            # suspect AF, §7.1). neurokit's delineation interpolates a P onset for
            # nearly every beat, so an onset-only ratio saturates at 1.0 even in AF
            # (where P waves are replaced by fibrillatory waves). Gate on the P-region
            # deflection instead: a peak-to-peak below ~15% of the R reference counts
            # as an absent/disorganized P.
            p_present = 0
            for i in range(n_beats):
                a, b = p_on[i], p_off[i]
                if np.isfinite(a) and np.isfinite(b) and 0 <= a < b < len(cleaned):
                    seg = cleaned[int(a):int(b)] - base
                    if seg.size and float(seg.max() - seg.min()) >= P_PRESENCE_FRACTION * r_ref:
                        p_present += 1
            m.p_wave_ratio = round(p_present / max(1, n_beats), 2)
            # P-wave duration/amplitude for atrial enlargement (P-mitrale/-pulmonale).
            pdur = [(p_off[i] - p_on[i]) for i in range(n_beats)
                    if np.isfinite(p_on[i]) and np.isfinite(p_off[i]) and p_off[i] > p_on[i]]
            m.p_wave_duration_ms = _median_ms(pdur, fs)
            # Global P and PR on the 12-lead median beat (D111), like 12SL/Glasgow: the
            # single-lead per-beat P is noisy (D103: P 28 ms short, SD 26; PR SD 15-21).
            if P_MEDIAN_BEAT and (m.p_wave_ratio or 0.0) >= 0.5:
                mb = _median_beat_p(sig.signal, rpeaks, qrs_on, fs, m.intervals.qt_ms)
                if mb is not None:
                    mb_pd, mb_pr = mb
                    if not P_MB_CONSENSUS or (m.p_wave_duration_ms is not None
                                              and abs(mb_pd - m.p_wave_duration_ms) <= 40.0):
                        m.p_wave_duration_ms = mb_pd
                    if m.intervals.pr_ms is not None and (
                            not P_MB_CONSENSUS or abs(mb_pr - m.intervals.pr_ms) <= 20.0):
                        m.intervals.pr_ms = mb_pr
            m.p_wave_amplitude_uv = _p_amplitude(sig, p_on, p_off)

        # ST measured off the envelope QRS offsets (J points), PR-segment baseline.
        _measure_st(sig, rpeaks, qrs_on, qrs_off, m)

        # Atrial activity independent of the QRS: flutter / AV dissociation (§7.1).
        atrial_leads = {ln: sig.lead(ln).astype(np.float64)
                        for ln in ("II", "V1") if ln in sig.lead_names}
        if not atrial_leads:
            atrial_leads = {sig.lead_names[lead_idx]: cleaned}
        pr_unstable = bool((m.p_onset_ratio or 0.0) >= 0.5 and m.pr_std_ms is not None
                           and m.pr_std_ms >= PR_STD_DISSOCIATION_MS)
        rate, av, notes = atrial_activity(
            atrial_leads, rpeaks, fs,
            heart_rate_bpm=m.heart_rate_bpm, rr_cv=m.rr_cv, r_ref=r_ref,
            pr_unstable=pr_unstable, qrs_on=qrs_on, qrs_off=qrs_off,
            # The spectral flutter search reads its own leads (II, III, aVF, V1): F waves are
            # best seen inferiorly; with II and V1 alone it missed flutter one lead short
            # of the two required (PTB-XL 15676, D83).
            spectral_leads={ln: sig.lead(ln).astype(np.float64)
                            for ln in FLUTTER_SPECTRAL_LEADS if ln in sig.lead_names} or None,
        )
        # AV dissociation means no fixed PR: when a P onset was found before most
        # beats and the PR is stable, the atria are conducting (sinus rhythm with a
        # long PR / long QT left a periodic residual) — not a complete block.
        if (av is True and (m.p_onset_ratio or 0.0) >= 0.5
                and m.pr_std_ms is not None and m.pr_std_ms < PR_STD_DISSOCIATION_MS):
            av = False
            notes.append(f"atrial activity: stable PR (std {m.pr_std_ms} ms) -> conducted, not dissociated")
        # A complete block has P waves by definition: without organised P waves the
        # periodic residual is fibrillatory/noise (AF with a slow response), not
        # dissociation.
        if (av is True and _lacks_organised_p(m.p_wave_ratio, n_beats)
                and (m.p_onset_ratio or 0.0) < 0.5):
            av = False
            notes.append("atrial activity: no organised P waves -> not AV dissociation")
        if ATRIAL_P_TRAIN:
            train = p_train({ln: sig.lead(ln).astype(np.float64)
                             for ln in P_TRAIN_LEADS if ln in sig.lead_names}, rpeaks, fs)
            flutter = rate is not None and FLUTTER_BAND_BPM[0] <= rate <= FLUTTER_BAND_BPM[1]
            av = p_train_dissociated(train, m.heart_rate_bpm) and not flutter
            if av:
                rate = round(train["rate_bpm"], 1)
                notes.append(f"atrial activity: P train {rate}/min, PR spread "
                             f"{round(1000 * train['pr_iqr_s'])} ms -> AV dissociation")
        m.atrial_rate_bpm, m.av_dissociation = rate, av
        m.fwave_index = fwave_index(atrial_leads, rpeaks, fs, qrs_on=qrs_on, qrs_off=qrs_off)
        m.notes.extend(notes)

    # --- QTc ---
    if m.intervals.qt_ms is not None and rr_s_mean:
        m.intervals.qtc_bazett_ms = round(qtc_bazett(m.intervals.qt_ms, rr_s_mean), 1)
        m.intervals.qtc_fridericia_ms = round(
            qtc_fridericia(m.intervals.qt_ms, rr_s_mean), 1
        )
        m.intervals.qtc_framingham_ms = round(
            qtc_framingham(m.intervals.qt_ms, rr_s_mean), 1
        )

    # --- Frontal QRS axis ---
    m.axes = Axes(qrs_deg=_qrs_axis(sig, rpeaks, fs, qrs_on, qrs_off),
                  p_deg=_p_axis(sig, rpeaks, fs, p_on, p_off) if p_on is not None and p_off is not None else None)

    # --- Sokolow-Lyon voltage (LVH) + the other hypertrophy voltages ---
    m.sokolow_lyon_uv = _sokolow_lyon(sig, rpeaks, fs)
    m.voltages = measure_voltages(sig, rpeaks, qrs_on, fs,
                                  {ld: _lead_mask(sig, ld, rpeaks, fs) for ld in sig.lead_names},
                                  exclude={ld for ld in sig.lead_names if _lead_off(sig, ld)})

    # --- Terminal QRS forces in V1/V6 (RSR' / broad-S typing for BBB) ---
    for lead, attr in (("V1", "qrs_v1_terminal_positive"), ("V6", "qrs_v6_terminal_positive")):
        if lead in sig.lead_names and not _lead_off(sig, lead):
            mk = _lead_mask(sig, lead, rpeaks, fs)
            setattr(m, attr, terminal_qrs_sign(sig.lead(lead), rpeaks[mk], qrs_off[mk], fs, qrs_on[mk]))

    # --- Limb-lead QRS amplitude (for low-voltage detection, TZ §7.5) ---
    m.qrs_amplitude_limb_uv = _limb_qrs_amplitude(sig, rpeaks, fs)

    # --- Morphology (peaked/flat/inverted T) ---
    m.morphology.st = m.morphology.st  # populated by _measure_st
    _measure_t_morphology(sig, rpeaks, fs, m)
    _measure_delta(sig, rpeaks, qrs_on, qrs_off, fs, m)

    return m


def _qrs_amplitude(row: np.ndarray, rpeaks: np.ndarray, fs: float) -> float:
    """Net QRS deflection = median signed peak amplitude at R locations (µV)."""
    if len(rpeaks) == 0:
        return 0.0
    win = int(0.05 * fs)
    vals = []
    for r in rpeaks:
        a, b = max(0, r - win), min(len(row), r + win)
        seg = row[a:b]
        # Signed extreme (largest absolute deflection keeps its sign).
        vals.append(seg[np.argmax(np.abs(seg))])
    return float(np.median(vals))


_LIMB_ANGLES_DEG = {"I": 0.0, "II": 60.0, "III": 120.0, "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}


def _qrs_net_deflection(sig: SignalRepresentation, lead: str, rpeaks: np.ndarray,
                        qrs_on: np.ndarray, qrs_off: np.ndarray, fs: float) -> float | None:
    """Median over beats of (max + min) of the QRS relative to the PR baseline (µV)."""
    row = sig.lead(lead)
    mk = _lead_mask(sig, lead, rpeaks, fs)
    b_lo, b_hi = int(0.05 * fs), int(0.01 * fs)
    vals = []
    for i in np.flatnonzero(mk):
        a, b = qrs_on[i], qrs_off[i]
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            continue
        a, b = int(a), int(b)
        base = float(np.median(row[max(0, a - b_lo):max(1, a - b_hi)]))
        seg = row[a:b] - base
        if seg.size:
            vals.append(float(seg.max() + seg.min()))
    return float(np.median(vals)) if vals else None


P_MEDIAN_BEAT = True       # D111: accepted on the held-out check (fold 10)
P_MB_FRACTION = 0.20       # P bounds: 20 % of the P peak above the window's 10th percentile
P_MB_CONSENSUS = 1         # use the median-beat value only near the per-beat one (D111)


def _median_beat_p(signal: np.ndarray, rpeaks: np.ndarray, qrs_on: np.ndarray, fs: float,
                   qt_ms: float | None) -> tuple[float, float] | None:
    """(P duration, PR) in ms from the 12-lead median beat (D111).

    Beats not premature (RR >= 0.8 x median) are aligned to the strip median beat by
    +-10 ms cross-correlation (r >= 0.9 kept); on their median, the P is searched from the
    previous T end (QRS onset + QT - RR) to 10 ms before the QRS onset: spatial magnitude
    above the straight line joining the window ends, bounded where it falls to
    P_MB_FRACTION of its peak above the window's 10th percentile. The QRS onset is the
    envelope onset (median offset to R), validated against 12SL (D103).
    """
    rp = np.asarray(rpeaks, dtype=int)
    if len(rp) < 4 or not np.isfinite(qrs_on).any():
        return None
    rr = np.diff(rp)
    med = float(np.median(rr))
    pre, post = int(min(0.5, 0.8 * med / fs) * fs), int(min(0.3, 0.6 * med / fs) * fs)
    sh = int(0.01 * fs)
    idx = [k for k in range(1, len(rp)) if rr[k - 1] >= 0.8 * med
           and rp[k] - pre - sh >= 0 and rp[k] + post + sh < signal.shape[1]]
    if len(idx) < 3:
        return None
    x = np.asarray(signal, dtype=np.float64)
    m0 = np.median(np.stack([x[:, rp[k] - pre:rp[k] + post] for k in idx]), axis=0)
    aligned = []
    for k in idx:
        best, bs = -2.0, 0
        for d in range(-sh, sh + 1):
            c = np.corrcoef(x[:, rp[k] - pre + d:rp[k] + post + d].ravel(), m0.ravel())[0, 1]
            if c > best:
                best, bs = c, d
        if best >= 0.9:
            aligned.append(x[:, rp[k] - pre + bs:rp[k] + post + bs])
    mb = np.median(np.stack(aligned), axis=0) if len(aligned) >= 3 else m0
    q = int(round(pre + float(np.nanmedian(qrs_on - rp))))
    t_end = (qt_ms / 1000.0 * fs) if qt_ms else 0.40 * fs
    lo = max(1, int(q + t_end - med + 0.02 * fs), int(q - 0.35 * fs))
    hi = q - int(0.01 * fs)
    if hi - lo < int(0.06 * fs):
        return None
    seg = mb[:, lo:hi]
    line = np.linspace(seg[:, :3].mean(axis=1), seg[:, -3:].mean(axis=1), seg.shape[1]).T
    mag = np.convolve(np.sqrt(((seg - line) ** 2).sum(axis=0)), np.ones(5) / 5, mode="same")
    pk = int(np.argmax(mag))
    base = float(np.percentile(mag, 10))
    thr = base + P_MB_FRACTION * (float(mag[pk]) - base)
    a = pk
    while a > 0 and mag[a] > thr:
        a -= 1
    b = pk
    while b < len(mag) - 1 and mag[b] > thr:
        b += 1
    if b - a < int(0.03 * fs):
        return None
    return (b - a) / fs * 1000.0, (q - (lo + a)) / fs * 1000.0


def _p_axis(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float,
            p_on: np.ndarray, p_off: np.ndarray) -> float | None:
    """Frontal P axis (degrees), D99.

    Per limb lead, the median over beats of the P-wave area above the straight line
    joining its onset and offset (a slope across the P — T wave tail, wander — is not
    taken for P), combined on the hexaxial frame like the QRS axis. Sinus P points
    down and to the left (0-75 deg); a low atrial or retrograde focus points up.
    """
    names = set(sig.lead_names)
    x = y = 0.0
    n = 0
    for lead, deg in _LIMB_ANGLES_DEG.items():
        if lead not in names or _lead_off(sig, lead):
            continue
        row = sig.lead(lead).astype(np.float64)
        vals = []
        for i in range(len(rpeaks)):
            a, b = p_on[i], p_off[i]
            if not (np.isfinite(a) and np.isfinite(b)) or b - a < int(0.02 * fs) or a < 0 or b >= row.size:
                continue
            seg = row[int(a):int(b) + 1]
            vals.append(float(np.mean(seg - np.linspace(seg[0], seg[-1], seg.size))))
        if len(vals) < 3:
            continue
        net = float(np.median(vals))
        x += net * np.cos(np.radians(deg))
        y += net * np.sin(np.radians(deg))
        n += 1
    if n < 2 or (abs(x) < 1e-6 and abs(y) < 1e-6):
        return None
    return round(float(np.degrees(np.arctan2(y, x))), 1)


def _qrs_axis(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float,
              qrs_on: np.ndarray | None = None, qrs_off: np.ndarray | None = None) -> float | None:
    """Frontal QRS axis (degrees).

    Net QRS deflection (R + S, i.e. max + min inside the QRS relative to the PR
    baseline) of all six limb leads, combined as a least-squares vector on the
    hexaxial frame — the conventional definition. Against the GE 12SL and
    Glasgow axes on PTB-XL (1691 records) this reads within 15° on 94 % of
    records (MAE 4.7°, the two commercial algorithms differ by 4.7° between
    themselves); the earlier largest-peak-in-I/aVF estimate managed 72 %
    (MAE 13°) (D49). Falls back to I/aVF peaks when no QRS bounds exist.
    """
    names = set(sig.lead_names)
    if len(rpeaks) == 0:
        return None
    if qrs_on is not None and qrs_off is not None and np.isfinite(qrs_on).any():
        x = y = 0.0
        n = 0
        for lead, deg in _LIMB_ANGLES_DEG.items():
            if lead not in names or _lead_off(sig, lead):
                continue
            net = _qrs_net_deflection(sig, lead, rpeaks, qrs_on, qrs_off, fs)
            if net is None:
                continue
            x += net * np.cos(np.radians(deg))
            y += net * np.sin(np.radians(deg))
            n += 1
        if n >= 2 and (abs(x) > 1e-6 or abs(y) > 1e-6):
            return round(float(np.degrees(np.arctan2(y, x))), 1)
    if not {"I", "aVF"}.issubset(names) or _lead_off(sig, "I") or _lead_off(sig, "aVF"):
        return None
    rp_i, rp_f = rpeaks[_lead_mask(sig, "I", rpeaks, fs)], rpeaks[_lead_mask(sig, "aVF", rpeaks, fs)]
    if len(rp_i) == 0 or len(rp_f) == 0:
        return None
    i_amp = _qrs_amplitude(sig.lead("I"), rp_i, fs)
    f_amp = _qrs_amplitude(sig.lead("aVF"), rp_f, fs)
    return round(float(np.degrees(np.arctan2(f_amp, i_amp))), 1)


def _sokolow_lyon(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    names = set(sig.lead_names)
    if not {"V1", "V5", "V6"}.issubset(names) or len(rpeaks) == 0:
        return None
    if any(_lead_off(sig, ld) for ld in ("V1", "V5", "V6")):
        return None
    win = int(0.05 * fs)

    def deep_s(lead: str) -> float:
        row = sig.lead(lead)
        vals = []
        for r in rpeaks[_lead_mask(sig, lead, rpeaks, fs)]:
            a, b = max(0, r - win), min(len(row), r + win)
            vals.append(abs(min(row[a:b])))
        return float(np.median(vals))

    def tall_r(lead: str) -> float:
        row = sig.lead(lead)
        vals = []
        for r in rpeaks[_lead_mask(sig, lead, rpeaks, fs)]:
            a, b = max(0, r - win), min(len(row), r + win)
            vals.append(max(row[a:b]))
        return float(np.median(vals))

    try:
        return round(deep_s("V1") + max(tall_r("V5"), tall_r("V6")), 1)
    except ValueError:  # a lead with no beats inside its window
        return None


def _p_amplitude(sig: SignalRepresentation, p_on: np.ndarray, p_off: np.ndarray) -> float | None:
    """Median P-wave amplitude in lead II (µV) — P-pulmonale if tall."""
    if "II" not in sig.lead_names:
        return None
    row = sig.lead("II")
    baseline = float(np.median(row))
    amps = []
    n = min(len(p_on), len(p_off))
    for i in range(n):
        a, b = p_on[i], p_off[i]
        if np.isfinite(a) and np.isfinite(b) and 0 <= a < b < len(row):
            seg = row[int(a):int(b)] - baseline
            if seg.size:
                amps.append(float(np.max(seg)))
    return round(float(np.median(amps)), 1) if amps else None


def _limb_qrs_amplitude(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    """Median QRS peak-to-peak across the limb leads I/II/III (µV, TZ §7.5).

    Per lead the median over its beats, then the median across leads. A pooled
    median over all (lead, beat) pairs is dominated by whichever lead has the
    most beats — on a 3x4 page the rhythm strip carries ten beats and I/III two
    each, so the "limb" amplitude became the lead-II amplitude (D52).
    """
    limb = [ln for ln in ("I", "II", "III") if ln in sig.lead_names and not _lead_off(sig, ln)]
    if not limb or len(rpeaks) == 0:
        return None
    win = int(0.05 * fs)
    per_lead = []
    for lead in limb:
        row = sig.lead(lead)
        amps = []
        for r in rpeaks[_lead_mask(sig, lead, rpeaks, fs)]:
            a, b = max(0, r - win), min(len(row), r + win)
            seg = row[a:b]
            amps.append(float(np.max(seg) - np.min(seg)))
        if amps:
            per_lead.append(float(np.median(amps)))
    return round(float(np.median(per_lead)), 1) if per_lead else None


def _measure_st(
    sig: SignalRepresentation,
    rpeaks: np.ndarray,
    qrs_on: np.ndarray,
    qrs_off: np.ndarray,
    m: Measurements,
) -> None:
    """ST deviation at J, J+60, J+80 ms per lead (TZ §6).

    J = envelope QRS offset. Baseline = median of the PR segment
    ([onset-50, onset-10] ms) per beat — a single sample is too noisy on real
    recordings and produced spurious ST calls. Also records the net QRS
    deflection per lead so Layer 4 can judge concordance (secondary ST-T changes).
    """
    fs = sig.sampling_rate_hz
    pairs = [(int(a), int(b)) for a, b in zip(qrs_on, qrs_off)
             if np.isfinite(a) and np.isfinite(b) and b > a]
    if not pairs:
        return
    s60, s80 = int(0.06 * fs), int(0.08 * fs)
    b_lo, b_hi = int(ST_BASELINE_MS[0] / 1000.0 * fs), int(ST_BASELINE_MS[1] / 1000.0 * fs)
    j_win = int(J_REFINE_MS / 1000.0 * fs)
    st_list: list[STMeasurement] = []
    all_pairs = [(int(a), int(b), i) for i, (a, b) in enumerate(zip(qrs_on, qrs_off))
                 if np.isfinite(a) and np.isfinite(b) and b > a]
    for li, lead in enumerate(sig.lead_names):
        row = sig.signal[li]
        if _lead_off(sig, lead):
            continue  # no ST measurement for a lead that was not recorded (D45)
        # Per-lead J refinement: the multi-lead envelope offset can land on the
        # upstroke of a deep S wave in V1-V3 (anterior MI), reading the J point on
        # the S wave; move J forward (<= J_REFINE_MS) to where this lead's slope has
        # flattened to J_REFINE_SLOPE_FRACTION of its own QRS slope.
        slope = np.abs(np.convolve(np.diff(row, prepend=row[0]), np.ones(max(1, int(0.005 * fs))) / max(1, int(0.005 * fs)), mode="same"))
        mk = _lead_mask(sig, lead, rpeaks, fs)
        lead_pairs = [(on, j) for on, j, i in all_pairs if mk[i]] or pairs
        j_vals, j60_vals, j80_vals, s_vals, r_vals = [], [], [], [], []
        for on, j in lead_pairs:
            lo, hi = max(0, on - b_lo), max(1, on - b_hi)
            baseline = float(np.median(row[lo:hi])) if hi > lo else float(row[max(0, on - 1)])
            if j > on:
                q = row[on:min(len(row), j + 1)]
                s_vals.append(baseline - float(np.min(q)))
                r_vals.append(float(np.max(q)) - baseline)
            if j_win > 0 and j + j_win < len(row) and j > on:
                qrs_slope = float(np.max(slope[on:j + 1]))
                flat = np.flatnonzero(slope[j:j + j_win] <= J_REFINE_SLOPE_FRACTION * qrs_slope)
                j = j + int(flat[0]) if flat.size else j + j_win
            if j < len(row):
                j_vals.append(row[j] - baseline)
            if j + s60 < len(row):
                j60_vals.append(row[j + s60] - baseline)
            if j + s80 < len(row):
                j80_vals.append(row[j + s80] - baseline)
        st_list.append(
            STMeasurement(
                lead=lead,
                j_point_uv=round(float(np.median(j_vals)), 1) if j_vals else None,
                j60_uv=round(float(np.median(j60_vals)), 1) if j60_vals else None,
                j80_uv=round(float(np.median(j80_vals)), 1) if j80_vals else None,
                qrs_net_uv=round(_qrs_amplitude(row, rpeaks[mk] if mk.any() else rpeaks, fs), 1),
                s_depth_uv=round(float(np.median(s_vals)), 1) if s_vals else None,
                r_height_uv=round(float(np.median(r_vals)), 1) if r_vals else None,
            )
        )
    m.morphology.st = st_list


DELTA_SLOPE_FRACTION = 0.5


def _measure_delta(sig: SignalRepresentation, rpeaks: np.ndarray, qrs_on: np.ndarray,
                   qrs_off: np.ndarray, fs: float, m: Measurements) -> None:
    """Delta-wave descriptor (D51): per lead, ms from the QRS onset until the
    smoothed |slope| first reaches DELTA_SLOPE_FRACTION of the beat's maximal QRS
    slope, median over beats; ``delta_ms`` = median over leads. On PTB-XL the
    median over leads is 54 ms (p10-p90 26-88) on WPW records and 24 ms (18-35)
    otherwise. Needs QRS bounds; leads without signal are skipped."""
    if len(rpeaks) < 3 or qrs_on is None or not np.isfinite(qrs_on).any():
        return
    k = max(1, int(0.004 * fs))
    out: dict[str, float] = {}
    for li, lead in enumerate(sig.lead_names):
        if _lead_off(sig, lead):
            continue
        row = sig.signal[li]
        slope = np.abs(np.convolve(np.diff(row, prepend=row[0]), np.ones(k) / k, mode="same"))
        mk = _lead_mask(sig, lead, rpeaks, fs)
        vals = []
        for i in np.flatnonzero(mk):
            a, b = qrs_on[i], qrs_off[i]
            if not (np.isfinite(a) and np.isfinite(b)) or b <= a + 5:
                continue
            a, b = int(a), int(b)
            s = slope[a:b]
            mx = float(s.max())
            if mx <= 0:
                continue
            idx = np.flatnonzero(s >= DELTA_SLOPE_FRACTION * mx)
            vals.append((int(idx[0]) if idx.size else b - a) / fs * 1000.0)
        if vals:
            out[lead] = round(float(np.median(vals)), 1)
    m.morphology.delta_ms_by_lead = out
    m.morphology.delta_ms = round(float(np.median(list(out.values()))), 1) if out else None


def _measure_t_morphology(
    sig: SignalRepresentation, rpeaks: np.ndarray, fs: float, m: Measurements
) -> None:
    """Measure the T wave per lead on the median beat (amplitude, QRS ratio,
    half-width, asymmetry) and classify flat / inverted T (TZ §6).

    The T window starts after the QRS (half the QRS width plus 40 ms past the R,
    at least 100 ms) and ends 200 ms before the next beat: the fixed 100-450 ms
    window reached the next QRS at fast rates and the S wave under a wide QRS,
    and 68 of 141 "peaked T" calls on the 664 set were QRS remnants 8-20 ms wide
    (D61). A deflection narrower than T_MIN_HALF_WIDTH_S is not a T wave and is
    neither inverted nor peaked. Peaked T itself is decided by the rules from the
    norm table (``t_wave``), on the descriptors stored in ``morphology.t_waves``."""
    if len(rpeaks) < 2:
        return
    rr = int(np.median(np.diff(rpeaks))) if len(rpeaks) >= 2 else int(0.8 * fs)
    qrs_ms = m.intervals.qrs_ms or 90.0
    t_start = max(int(0.10 * fs), int((qrs_ms / 2000.0 + 0.04) * fs))
    t_end = min(int(0.45 * fs), rr - int(T_NEXT_BEAT_GUARD_S * fs))
    if t_end - t_start < int(0.08 * fs):
        t_end = min(int(0.45 * fs), rr - 1)          # very fast rhythm: keep what room there is
    min_hw = int(T_MIN_HALF_WIDTH_S * fs)
    for li, lead in enumerate(sig.lead_names):
        row = sig.signal[li]
        if _lead_off(sig, lead):
            continue  # an unrecorded lead is not a flat T wave (D45)
        rp = rpeaks[_lead_mask(sig, lead, rpeaks, fs)]
        if len(rp) == 0:
            continue
        qrs_amp = abs(_qrs_amplitude(row, rp, fs)) + 1e-6
        base = float(np.median(row))
        segs = [row[r + t_start:r + t_end] - base for r in rp if r + t_end < len(row) and t_end > t_start]
        if not segs:
            continue
        length = min(len(s) for s in segs)
        med = np.median(np.stack([s[:length] for s in segs]), axis=0)
        peak = int(np.argmax(np.abs(med)))
        t_amp = float(med[peak])
        ratio = t_amp / qrs_amp
        # Half-amplitude width and up/down-slope asymmetry of the median beat's T.
        above = np.abs(med) >= 0.5 * abs(t_amp)
        lo = peak
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak
        while hi < length - 1 and above[hi + 1]:
            hi += 1
        half_width = hi - lo + 1
        asym = (peak - lo + 0.5) / (hi - peak + 0.5)
        is_t = half_width >= min_hw
        m.morphology.t_waves.append(TWave(
            lead=lead, amplitude_uv=round(t_amp, 1), qrs_ratio=round(ratio, 3),
            half_width_ms=round(half_width / fs * 1000.0, 1) if is_t else None,
            asymmetry=round(asym, 2) if is_t else None))
        if is_t and t_amp < 0 and abs(t_amp) > 100.0:
            m.morphology.inverted_t_leads.append(lead)
        elif abs(t_amp) < 100.0 and abs(ratio) < 0.15:
            # "Flat" needs to be small in absolute AND relative terms: a 100 uV
            # absolute cut alone flagged 70 % of children (small limb-lead
            # voltages) (D44).
            m.morphology.flat_t_leads.append(lead)


def _qrs_partner_consistency(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    """Median over beats of each beat's best correlation to another beat (D89).

    The single-template consistency drops when a strip has two or more stable QRS
    shapes (PVCs, bigeminy, intermittent pacing): the median template blends them
    (PTB-XL 17939, RBBB with PVCs: 0.37). Every beat there still has a near-identical
    twin (0.997); noise does not repeat itself. Same lead and window as the template
    measure. ``None`` with fewer than three beats.
    """
    if len(rpeaks) < 3:
        return None
    li = sig.lead_names.index("II") if "II" in sig.lead_names else 0
    row = sig.signal[li].astype(np.float64)
    w = int(0.09 * fs)
    beats = [row[int(r) - w:int(r) + w] for r in rpeaks
             if int(r) - w >= 0 and int(r) + w < len(row)]
    if len(beats) < 3:
        return None
    m = np.vstack(beats)
    if np.any(np.std(m, axis=1) < 1e-9):
        return 0.0
    c = np.corrcoef(m)
    np.fill_diagonal(c, -1.0)
    return round(float(np.median(c.max(axis=1))), 3)


def _qrs_template_consistency(
    sig: SignalRepresentation, rpeaks: np.ndarray, fs: float,
) -> float | None:
    """Median correlation of each QRS complex to the median beat, in [0,1] (§5).

    Real ECG has a self-consistent QRS shape from beat to beat — even in atrial
    fibrillation, where timing is irregular but morphology is stable — so this
    stays high (~0.98). Band-limited noise and artifact have no repeatable beat,
    so spurious "R-peaks" correlate poorly (~0.6-0.75). It is the organization
    signal a plain SNR/HF quality index misses once the band-pass has smoothed
    broadband noise into the ECG band. ``None`` if too few beats to judge.
    """
    if len(rpeaks) < 3:
        return None
    li = sig.lead_names.index("II") if "II" in sig.lead_names else 0
    row = sig.signal[li].astype(np.float64)
    # +-90 ms captures the full QRS even for wide (BBB/paced) complexes, and the
    # wider window also drives noise correlation down (a random signal decorrelates
    # over a longer span) — sharpening the separation from real ECG.
    w = int(0.09 * fs)
    beats = [row[int(r) - w:int(r) + w] for r in rpeaks
             if int(r) - w >= 0 and int(r) + w < len(row)]
    if len(beats) < 3:
        return None
    m = np.vstack(beats)
    templ = np.median(m, axis=0)
    if np.std(templ) < 1e-9:
        return 0.0
    cors = [float(np.corrcoef(b, templ)[0, 1]) if np.std(b) > 1e-9 else 0.0 for b in m]
    return round(float(np.median(cors)), 3)
