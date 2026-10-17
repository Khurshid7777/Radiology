"""Trace vectorisation: ink mask -> one y(x) per panel (TZ §5.7).

Per column the ink in the panel's search band splits into vertical runs. A Viterbi
pass picks one run per column (or an explicit *gap*) minimising

* the vertical gap between consecutive runs (a drawn trace is continuous: a steep
  QRS edge is a tall run touching its neighbours, not a jump),
* a weak pull towards the panel's own baseline (a neighbour row's deep S wave
  entering the band loses, even where it is nearer),
* a fixed cost per gap column (missing ink stays missing: NaN, TZ §7).

The value read from the chosen run depends on its shape — two methods in one pass:

* **weighted centroid** (sub-pixel) on thin runs: flat and slow segments;
* **path end** on tall runs: at a local extreme (R peak, S trough) the run's far end
  minus half a stroke, so peaks are not flattened by averaging (the column-mean
  failure the TZ warns about); on a monotonic edge the run's middle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

START_JUMP_WEIGHT = 0.25


@dataclass
class PanelTrace:
    y: np.ndarray              # float [n_cols], NaN where no trace
    ambiguity: float           # share of columns where another run competed closely
    edge_contact: float        # share of chosen runs touching the search band edge (a neighbour's ink?)
    stroke_px: float


def column_runs(col: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of True runs in a 1-D bool array."""
    if not col.any():
        return []
    d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def estimate_stroke(ink: np.ndarray) -> float:
    """Typical stroke thickness (px): median height of thin runs (flat segments dominate)."""
    hs = []
    for x in range(0, ink.shape[1], max(1, ink.shape[1] // 400)):
        hs += [b - a for a, b in column_runs(ink[:, x])]
    hs = np.array([h for h in hs if h > 0])
    if hs.size == 0:
        return 2.0
    return float(max(1.0, np.median(hs[hs <= np.percentile(hs, 60)])))


def trace_panel(ink: np.ndarray, darkness: np.ndarray, x0: int, x1: int, y0: int, y1: int, baseline: float, *,
                stroke_px: float, pitch_px: float, baseline_weight: float = 0.6, gap_cost: float = 2.5,
                compete_margin: float = 1.0, skeleton: np.ndarray | None = None,
                other_baselines: list[float] | None = None, neighbour_cost: float = 4.0) -> PanelTrace:
    """Follow one trace through columns [x0, x1) inside the band rows [y0, y1).

    With ``skeleton`` the runs (presence, vertical extent, extremes) come from the 1-px
    skeleton and only the sub-pixel centroid of thin runs from the raw ink around it.
    """
    raw_ink = ink
    if skeleton is not None:
        ink = skeleton
    h, w = ink.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    n = max(0, x1 - x0)
    if n == 0:
        return PanelTrace(np.zeros(0), 1.0, 0.0, stroke_px)
    sub = ink[y0:y1, x0:x1]
    runs = [[(a + y0, b + y0) for a, b in column_runs(sub[:, i])] for i in range(n)]
    s = max(1.0, stroke_px)
    INF = 1e18
    # state 0 = gap; states 1.. = runs. cost[i][k], back[i][k]; gap keeps the last run's centre.
    costs: list[np.ndarray] = []
    backs: list[np.ndarray] = []
    last_c: list[np.ndarray] = []       # centre carried by each state (gap: last seen run centre)
    competing = 0
    for i in range(n):
        rs = runs[i]
        k = len(rs) + 1
        c = np.full(k, INF)
        b = np.zeros(k, dtype=np.int64)
        lc = np.zeros(k)
        centres = np.array([(a + e - 1) / 2.0 for a, e in rs]) if rs else np.zeros(0)
        own = baseline_weight * np.abs(centres - baseline) / max(pitch_px, 1.0) * 4.0 if rs else np.zeros(0)
        if rs and other_baselines:
            # A thin (flat) run lying on another row's baseline is that row's trace: riding it
            # costs more than a gap. A tall run (this row's R or S crossing over) is not penalised.
            heights = np.array([e - a for a, e in rs])
            near = np.min(np.abs(centres[:, None] - np.asarray(other_baselines)[None, :]), axis=1)
            own = own + np.where((near < 0.2 * pitch_px) & (heights <= 3.0 * s), neighbour_cost, 0.0)
        if i == 0:
            c[0] = gap_cost
            lc[0] = baseline
            if rs:
                # A trace starts from its baseline: entering far from it costs like a jump
                # (the first columns otherwise latch onto header text or a neighbour).
                c[1:] = own + START_JUMP_WEIGHT * np.abs(centres - baseline) / s
                lc[1:] = centres
        else:
            pc, plc = costs[-1], last_c[-1]
            prs = runs[i - 1]
            # to gap
            j = int(np.argmin(pc))
            c[0], b[0], lc[0] = pc[j] + gap_cost, j, plc[j]
            for r, (a, e) in enumerate(rs):
                best, arg = INF, 0
                # from gap: distance to the carried centre
                d0 = abs((a + e - 1) / 2.0 - plc[0]) / s
                cand = pc[0] + d0
                if cand < best:
                    best, arg = cand, 0
                for q, (pa, pe) in enumerate(prs):
                    gap = max(0, a - pe, pa - e) / s
                    cand = pc[q + 1] + gap
                    if cand < best:
                        best, arg = cand, q + 1
                c[r + 1] = best + own[r]
                b[r + 1] = arg
                lc[r + 1] = centres[r]
            if len(rs) > 1:
                srt = np.sort(c[1:])
                if srt[1] - srt[0] < compete_margin:
                    competing += 1
        costs.append(c)
        backs.append(b)
        last_c.append(lc)
    # Backtrack.
    choice = np.zeros(n, dtype=np.int64)
    choice[-1] = int(np.argmin(costs[-1]))
    for i in range(n - 1, 0, -1):
        choice[i - 1] = backs[i][choice[i]]
    chosen = [runs[i][choice[i] - 1] if choice[i] > 0 else None for i in range(n)]
    tops = np.array([np.nan if r is None else float(r[0]) for r in chosen])
    bots = np.array([np.nan if r is None else float(r[1] - 1) for r in chosen])
    y = np.full(n, np.nan)
    spikes = _spike_columns(raw_ink, chosen, x0, y0, y1, s) if skeleton is not None else {}
    touch = 0
    valid = 0
    for i, r in enumerate(chosen):
        if r is None:
            continue
        valid += 1
        a, e = r
        if a <= y0 or e >= y1:
            touch += 1
        if i in spikes:
            y[i] = spikes[i]
            continue
        thin_run = (e - a <= 2) if skeleton is not None else (e - a <= 2.0 * s)
        if thin_run:
            ra, re = a, e
            if skeleton is not None:      # the raw ink run around the skeleton pixel
                col = raw_ink[:, x0 + i]
                while ra > 0 and col[ra - 1] and a - ra < 2 * s:
                    ra -= 1
                while re < h and col[re] and re - e < 2 * s:
                    re += 1
            wts = darkness[ra:re, x0 + i].astype(np.float64) + 1e-6
            y[i] = float(np.sum(np.arange(ra, re) * wts) / np.sum(wts))
            continue
        # Extremes by the neighbours' run *ends*, not centres: an edge column next to the
        # peak column has the peak column's top above its own, so it stays an edge sample.
        pt, nt = _neighbour(tops, i, -1), _neighbour(tops, i, +1)
        pb, nb = _neighbour(bots, i, -1), _neighbour(bots, i, +1)
        # At an apex the skeleton stops short of the tip of a thick spike: read the raw
        # stroke's extreme in this column and step back half a stroke (D47 did the same).
        ra, re = a, e
        if skeleton is not None:
            col = raw_ink[:, x0 + i]
            reach = int(np.ceil(2 * s))     # the tip is at most ~a stroke beyond the skeleton end;
            while ra > 0 and col[ra - 1] and a - ra < reach:      # further up is a label or text
                ra -= 1
            while re < h and col[re] and re - e < reach:
                re += 1
        if pt is not None and nt is not None and a < pt and a <= nt:
            y[i] = ra + s / 2.0            # local maximum of the signal (top of the stroke)
        elif pb is not None and nb is not None and e - 1 > pb and e - 1 >= nb:
            y[i] = re - 1 - s / 2.0        # local minimum
        else:
            y[i] = (a + e - 1) / 2.0
    return PanelTrace(y, competing / max(1, n), touch / max(1, valid), s)


def _spike_columns(raw_ink: np.ndarray, chosen: list, x0: int, y0: int, y1: int, s: float) -> dict[int, float]:
    """A narrow spike drawn with a thick pen is a solid wedge whose skeleton breaks up, and
    the path then runs along its base (a 0.87 mV pediatric R was read as 0.06 mV).

    A column is in such a wedge when the raw stroke around the chosen skeleton run reaches
    far beyond it on one side (a stroke end lies about half a stroke beyond the skeleton)
    and beyond the path of the neighbouring columns.
    Near the wedge's extreme (within a stroke of the farthest tip around) the sample is the
    tip, half a stroke back; on its flanks the column is crossed by the steep edge, and the
    middle of the raw run is the sample. Returns {column: y}."""
    tips: dict[int, tuple[int, float, float]] = {}
    k = int(np.ceil(s))
    tops = np.array([np.inf if r is None else r[0] for r in chosen])
    bots = np.array([-np.inf if r is None else r[1] for r in chosen])
    for i, r in enumerate(chosen):
        if r is None:
            continue
        a, e = r
        col = raw_ink[:, x0 + i]
        ra, re = a, e
        while ra > y0 and col[ra - 1]:
            ra -= 1
        while re < y1 and col[re]:
            re += 1
        up, down = a - ra, re - e
        far = max(up, down)
        if far <= 3 * s or far < 3 * min(up, down):
            continue
        sign = -1 if up > down else 1
        # On a steep thin edge the stroke's own width makes the column cut a long run, but
        # the path in the next columns already covers it: only an unexplained reach counts.
        lo, hi = max(0, i - k), i + k + 1
        if (sign < 0 and ra >= tops[lo:hi].min() - 2 * s) or (sign > 0 and re <= bots[lo:hi].max() + 2 * s):
            continue
        tips[i] = (sign, ra + s / 2.0 if sign < 0 else re - 1 - s / 2.0, (ra + re - 1) / 2.0)
    out = {}
    for i, (sign, tip, mid) in tips.items():
        near = [t for j, (sg, t, _m) in tips.items() if abs(j - i) <= 2 * k and sg == sign]
        extreme = min(near) if sign < 0 else max(near)
        out[i] = tip if abs(tip - extreme) <= s else mid
    return out


def _neighbour(centres: np.ndarray, i: int, step: int) -> float | None:
    j = i + step
    while 0 <= j < len(centres):
        if np.isfinite(centres[j]):
            return float(centres[j])
        j += step
    return None


def resample_nan(values: np.ndarray, fs_in: float, fs_out: float, t_len_s: float) -> np.ndarray:
    """Linear resampling that never bridges a gap: an output sample is NaN unless both
    neighbouring input samples are valid."""
    n_out = int(round(t_len_s * fs_out))
    if values.size == 0 or n_out <= 0:
        return np.full(max(n_out, 0), np.nan)
    t = np.arange(n_out) / fs_out * fs_in
    i0 = np.clip(np.floor(t).astype(int), 0, len(values) - 1)
    i1 = np.clip(i0 + 1, 0, len(values) - 1)
    f = t - i0
    out = values[i0] * (1 - f) + values[i1] * f
    out[~(np.isfinite(values[i0]) & np.isfinite(values[i1]))] = np.nan
    return out
