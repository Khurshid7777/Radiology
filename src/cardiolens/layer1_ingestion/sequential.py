"""Three-channel strips: three rows, lead groups one after the other (D125).

Narrow-paper machines (MEDIUS, iMAC 300 and the like) print three leads at a time:
I/II/III, then aVR/aVL/aVF, then V1-V3, then V4-V6, each group a few seconds wide
with its labels in the gap before it. A phone photo shows the strip on a desk or a
carpet, often turned, and usually two or three of the four groups.

1. the paper is where the red grid is dense (a cream carpet or a grey cloth is as
   bright as paper, but carries no grid); the strip is turned so its long side is
   horizontal;
2. the three rows are read with the strip reader (``digitize_strip``);
3. group boundaries are columns where the rows carry no ink near their baselines
   (labels are glyph-sized and do not count); a group shorter than
   ``SEQ_MIN_GROUP_S`` is a cut-off end;
4. which group is which: the limb group satisfies Einthoven (II = I + III), the
   augmented group sums to zero; the groups follow the printing order;
5. of the two (or four) turns, the one whose groups match the leads best is kept.

Every group is its own time window (the groups need not be contiguous in time), so
the record carries ``discontinuities_s`` between them, as joined screen pages do."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import STANDARD_12_LEADS, Provenance, SignalRepresentation
from cardiolens.layer1_ingestion.strip import SCALE_CONFLICT_NOTE

SEQ_GROUPS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
SEQ_MIN_ASPECT = 2.2         # paper long side / short side at least this: a strip, not a page
SEQ_GRID_DENSITY = 0.04      # blurred fraction of red-grid pixels on paper (cloth and carpet: ~0)
SEQ_PAPER_BRIGHT = 0.7       # ... and of bright, weakly saturated pixels
SEQ_WHITE_V = 150            # grey-grid fallback (D141): near-white pixels ...
SEQ_WHITE_MAX_S = 60
SEQ_WHITE_FILL = 0.85        # ... filling this much of the window = paper
SEQ_GRID_LINE_RATIO = 8.0    # ... with a grid line this far above the spectrum's median
SEQ_GAP_MM = 6.0             # an inkless run at least this long ...
SEQ_GAP_ROWS = 2             # ... in at least this many rows at the same place is a group boundary
SEQ_MIN_GROUP_S = 1.2        # a shorter group is the cut-off end of the strip
SEQ_LIMB_MIN_R = 0.6         # |rank r(II, I + III)| above this: a chest group is not expected to look like that
SEQ_NAME_MIN_R = 0.5         # |r| a group must reach for its pattern to name the strip's leads (D138)
SEQ_POLARITY_WEIGHT = 0.3    # match score for QRS pointing the expected way (II up, aVR down)
SEQ_PULSE_WEIGHT = 0.5       # match score for "limb group first" when a pulse is found at the left
SEQ_INK_RATIOS = (0.7, 0.62)  # ink thresholds tried (a pale grid passes the strip reader's 0.78)
SEQ_EDGE_S = 0.1             # trimmed off both ends of a group (the pen settling, a label's tail)


def find_paper(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """The paper cut out of the photo and turned to the axes of its own rectangle, and the
    box ``(x0, y0, x1, y1)`` it came from; None when no gridded region is found."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    k = max(31, int(0.03 * max(bgr.shape[:2]))) | 1
    red = (((hue < 12) | (hue > 160)) & (sat > 25) & (val > 120)).astype(np.float32)
    bright = ((val > 140) & (sat < 90)).astype(np.float32)
    paper = ((cv2.blur(red, (k, k)) > SEQ_GRID_DENSITY) & (cv2.blur(bright, (k, k)) > SEQ_PAPER_BRIGHT))
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8))
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n >= 2 else 0
    if n < 2 or st[i, cv2.CC_STAT_AREA] < 0.05 * paper.size:
        # A compressed photo in a cold light shows the grid grey, not red (D141: FK strip at
        # ~5 px/mm, grid saturation ~20). The paper is then the largest near-white region,
        # if it carries a periodic grid.
        white = ((val > SEQ_WHITE_V) & (sat < SEQ_WHITE_MAX_S)).astype(np.float32)
        paper = cv2.blur(white, (k, k)) > SEQ_WHITE_FILL
        n, lab, st, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8))
        if n < 2:
            return None
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = st[i, :4]
        if st[i, cv2.CC_STAT_AREA] < 0.05 * paper.size or not _has_grid(cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)):
            return None
    pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
    (cx, cy), _size, ang = cv2.minAreaRect(pts)
    # Turn by the rectangle's small residual angle only; quarter turns are chosen later.
    if ang > 45:
        ang -= 90
    elif ang < -45:
        ang += 90
    rot = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    turned = cv2.warpAffine(bgr, rot, (bgr.shape[1], bgr.shape[0]), borderValue=(0, 0, 0))
    mask = cv2.warpAffine((lab == i).astype(np.uint8), rot, (bgr.shape[1], bgr.shape[0]))
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return turned[y0:y1, x0:x1], (x0, y0, x1, y1)


def _has_grid(gray: np.ndarray) -> bool:
    """A periodic fine grid: the column profile's strongest line in the 1-mm period range
    (3-15 px) stands well above the spectrum's median there (a white wall or a sheet of
    plain paper has no such line)."""
    prof = gray.astype(np.float64).mean(axis=0)
    if prof.size < 64:
        return False
    f = np.abs(np.fft.rfft((prof - prof.mean()) * np.hanning(prof.size)))
    fr = np.fft.rfftfreq(prof.size)
    band = (fr >= 1 / 15.0) & (fr <= 1 / 3.0)
    return bool(band.any() and f[band].max() >= SEQ_GRID_LINE_RATIO * np.median(f[band]))


def is_sequential_strip(bgr: np.ndarray) -> bool:
    found = find_paper(bgr)
    if found is None:
        return False
    h, w = found[0].shape[:2]
    return max(h, w) >= SEQ_MIN_ASPECT * min(h, w)


def _group_bounds(near: np.ndarray, px_per_mm: float, px_per_s: float) -> list[tuple[int, int]]:
    """``(x0, x1)`` of every group, in the coordinates of ``near`` (rows x columns)."""
    gap = max(2, int(SEQ_GAP_MM * px_per_mm))
    empty = np.zeros(near.shape[1], dtype=int)
    for row in near:
        z = ~row
        # Mark the columns of inkless runs at least ``gap`` long.
        run = np.zeros_like(z)
        idx = np.flatnonzero(z)
        if idx.size:
            for part in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if part.size >= gap:
                    run[part] = True
        empty += run
    sep = empty >= min(SEQ_GAP_ROWS, near.shape[0])
    groups, start = [], None
    for x, s in enumerate(np.append(sep, True)):
        if not s and start is None:
            start = x
        elif s and start is not None:
            if x - start >= SEQ_MIN_GROUP_S * px_per_s:
                groups.append((start, x))
            start = None
    return groups


def _rank_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation: one tracker jump to the next row must not decide the match."""
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def _limb_r(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """r(II, I + III) when the rows are I, II, III."""
    return _rank_r(b, a + c)


def _aug_r(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """r(aVL, -(aVR + aVF)) when the rows are aVR, aVL, aVF (they sum to zero)."""
    return _rank_r(b, -(a + c))


def _polarity(x: np.ndarray) -> float:
    """+1 when the trace's largest deflections point up (R-dominant), -1 when down."""
    med = float(np.median(x))
    up, down = float(np.percentile(x, 99.5)) - med, med - float(np.percentile(x, 0.5))
    return float(np.sign(up - down))


def _assign(groups: list[list[np.ndarray]], pulse: bool = False) -> tuple[int, float]:
    """Index into SEQ_GROUPS of the first group, and the match score of that choice.
    ``pulse``: a calibration pulse precedes the first group, i.e. the strip's start is on
    the photo and the first group is I, II, III."""
    limb = [_limb_r(*g) for g in groups]
    aug = [_aug_r(*g) for g in groups]
    best, best_score = 0, -np.inf
    for k in range(len(SEQ_GROUPS) - len(groups) + 1):
        score = 0.0
        for j in range(len(groups)):
            kind = k + j
            # A strip turned by 180 degrees has its groups in reverse order and every
            # row negated and reversed; Einthoven alone can then fit a wrong order, so
            # the polarity of QRS counts too: II points up and aVR down in nearly every record.
            if kind == 0:
                score += limb[j] + SEQ_POLARITY_WEIGHT * _polarity(groups[j][1])
            elif kind == 1:
                score += aug[j] - SEQ_POLARITY_WEIGHT * _polarity(groups[j][0])
            else:   # a chest group that looks like a limb or augmented one counts against
                score -= max(0.0, abs(limb[j]) - SEQ_LIMB_MIN_R)
        if pulse and k == 0:
            score += SEQ_PULSE_WEIGHT
        if score > best_score:
            best, best_score = k, score
    return best, best_score


def digitize_sequential(path: str | Path, *, mm_per_s: float = 25.0, mm_per_mv: float | None = None,
                        ink_ratios: tuple[float, ...] = SEQ_INK_RATIOS) -> list[dict[str, np.ndarray]]:
    """One ``{lead: trace_uv}`` per group, in time order; ``digitize_sequential.last_calibration``
    holds the provenance (``fs``, the groups' leads, the turn)."""
    import cv2

    from cardiolens.layer1_ingestion.strip import digitize_strip

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    found = find_paper(bgr)
    if found is None:
        raise ValueError("three-channel strip: no gridded paper found")
    paper = found[0]
    turns = (0, 2) if paper.shape[1] >= paper.shape[0] else (1, 3)
    best = None
    for t, ink_ratio in ((t, r) for t in turns for r in ink_ratios):
        view = np.ascontiguousarray(np.rot90(paper, -t)) if t else paper
        try:
            traces = digitize_strip(view, rows=3, lead_order=["r0", "r1", "r2"], mm_per_s=mm_per_s,
                                    mm_per_mv=mm_per_mv, ink_ratio=ink_ratio)
        except ValueError:
            continue
        calib = dict(digitize_strip.last_calibration)  # type: ignore[attr-defined]
        near = digitize_strip.last_near_ink  # type: ignore[attr-defined]
        ppm = calib["px_per_mm"]
        bounds = _group_bounds(near, ppm, ppm * mm_per_s)
        if not bounds:
            continue
        edge = int(SEQ_EDGE_S * ppm * mm_per_s)
        bounds = [(x0 + edge, x1 - edge) for x0, x1 in bounds]
        groups = [[traces[r][x0:x1] for r in ("r0", "r1", "r2")] for x0, x1 in bounds]
        first, score = _assign(groups, pulse=calib.get("pulses_found", 0) > 0)
        calib["ink_ratio"] = ink_ratio
        if best is None or score > best[0]:
            best = (score, t, first, groups, bounds, calib,
                    float(digitize_strip.last_coverage))  # type: ignore[attr-defined]
    if best is None:
        raise ValueError("three-channel strip: no three-row groups found")
    score, t, first, groups, bounds, calib, cov = best
    # Naming needs evidence (D138). Chest groups mimic the limb pattern (V2 tracks V1 + V3,
    # rank r up to 0.89 on synthetic records) but not the augmented one (aVL = -(aVR + aVF)).
    # So the leads are named only when the aVR-aVF group is confirmed where it was placed,
    # or the strip's start (a calibration pulse) is on the photo and its first group looks
    # like I-III. A MEDIUS strip of V1-V6 alone, turned and without a pulse, was named
    # I-III / aVR-aVF (match 0.42) — the scanner branch refuses such a page (SD-059).
    aug_ok = any(first + j == 1 and _limb_r(*g) <= -SEQ_NAME_MIN_R for j, g in enumerate(groups))
    start_ok = (first == 0 and calib.get("pulses_found", 0) > 0 and _limb_r(*groups[0]) >= SEQ_NAME_MIN_R)
    if not (aug_ok or start_ok):
        raise ValueError("three-channel strip: the limb groups are not recognised (neither the aVR-aVF group "
                         "nor the strip's start with its calibration pulse is on the photo) — the leads cannot "
                         "be named; photograph the whole strip")
    out = []
    for j, g in enumerate(groups):
        leads = SEQ_GROUPS[first + j]
        out.append({ln: (tr - float(np.median(tr))).astype(np.float32) for ln, tr in zip(leads, g, strict=True)})
    calib.update(layout="3-channel sequential", rotated_deg=90 * t, groups=[list(o) for o in out],
                 group_px=[list(b) for b in bounds], match_score=round(score, 3),
                 fs=float(calib["px_per_mm"] * mm_per_s), mm_per_s=mm_per_s)
    digitize_sequential.last_calibration = calib  # type: ignore[attr-defined]
    digitize_sequential.last_coverage = cov  # type: ignore[attr-defined]
    return out


def digitize_sequential_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """12-lead SignalRepresentation: each group valid in its own window, one after the other,
    with ``discontinuities_s`` between them; leads not on the photo stay out of
    ``lead_windows``."""
    groups = digitize_sequential(path, **kw)
    calib = digitize_sequential.last_calibration  # type: ignore[attr-defined]
    fs = calib["fs"]
    lengths = [min(len(v) for v in g.values()) for g in groups]
    signal = np.zeros((12, sum(lengths)), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    cuts: list[float] = []
    start = 0
    for g, n in zip(groups, lengths, strict=True):
        if start:
            cuts.append(start / fs)
        for ln, tr in g.items():
            signal[STANDARD_12_LEADS.index(ln), start:start + n] = tr[:n]
            windows[ln] = (start / fs, (start + n) / fs)
        start += n
    leads = [ln for g in groups for ln in g]
    note = (f"three-channel strip photo: {len(groups)} group(s), leads {', '.join(leads)} only; "
            f"{calib['mm_per_s']} mm/s assumed, groups read as separate windows")
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS), duration_s=start / fs,
        source="image",
        provenance=Provenance(digitized=True, confidence=round(digitize_sequential.last_coverage, 3),  # type: ignore[attr-defined]
                              lead_windows=windows, discontinuities_s=cuts,
                              notes=[f"calibration: {calib}", note]
                                    + ([SCALE_CONFLICT_NOTE] if calib.get("scale_conflict") else [])))


def render_sequential_strip(sig: SignalRepresentation, path: str | Path, *, groups: int = 2, first: int = 0,
                            seconds: float = 4.0, px_per_mm: int = 8, mm_per_s: float = 25.0,
                            mm_per_mv: float = 10.0, row_pitch_mm: float = 22.0, turn: int = 0) -> str:
    """A synthetic photo of a three-channel strip (tests): pink gridded paper on a dark,
    mottled desk, ``groups`` lead groups from ``SEQ_GROUPS[first]`` on, each ``seconds``
    long (group ``j`` shows the record's ``[j * seconds, (j + 1) * seconds)``), labels
    and a gap between them, a 1-mV pulse at the left; ``turn`` quarter turns clockwise."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import _draw_trace

    fs = sig.sampling_rate_hz
    gap = int(12 * px_per_mm)
    col = int(seconds * mm_per_s * px_per_mm)
    left = int(15 * px_per_mm)
    w = left + groups * (col + gap)
    pitch = int(row_pitch_mm * px_per_mm)
    h = pitch * 4
    paper = np.full((h, w, 3), (232, 228, 248), np.uint8)
    for x in range(0, w, px_per_mm):
        cv2.line(paper, (x, 0), (x, h), (205, 190, 245) if x % (5 * px_per_mm) else (150, 120, 235), 1)
    for y in range(0, h, px_per_mm):
        cv2.line(paper, (0, y), (w, y), (205, 190, 245) if y % (5 * px_per_mm) else (150, 120, 235), 1)
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    pulse_top = int(1000 * px_per_uv)
    yc = 2 * pitch
    if first == 0:   # the pulse is printed at the start of the strip
        pts = np.array([[4, 0], [6, 0], [6, -1], [11, -1], [11, 0], [13, 0]], np.float64)
        pts = np.column_stack([pts[:, 0] * px_per_mm, yc + pts[:, 1] * pulse_top]).astype(np.int32)
        cv2.polylines(paper, [pts.reshape(-1, 1, 2)], False, (30, 30, 30), 2)
    n = int(seconds * fs)
    for j in range(groups):
        x0 = left + j * (col + gap) + gap
        for r, ln in enumerate(SEQ_GROUPS[first + j]):
            y = (r + 1) * pitch
            cv2.putText(paper, ln, (x0 - gap + px_per_mm, y - px_per_mm), cv2.FONT_HERSHEY_SIMPLEX,
                        0.05 * px_per_mm, (40, 40, 40), 1, cv2.LINE_AA)
            seg = sig.lead(ln)[j * n:(j + 1) * n]
            _draw_trace(paper, seg, fs, x0, y, mm_per_s * px_per_mm, px_per_uv)
    rng = np.random.default_rng(0)
    margin = int(0.25 * h)
    desk = rng.integers(20, 90, size=((h + 2 * margin) // 16 + 1, (w + 2 * margin) // 16 + 1, 3)).astype(np.uint8)
    img = cv2.resize(desk, ((w + 2 * margin) // 16 * 16 + 16, (h + 2 * margin) // 16 * 16 + 16),
                     interpolation=cv2.INTER_NEAREST)[:h + 2 * margin, :w + 2 * margin].copy()
    img[margin:margin + h, margin:margin + w] = paper
    for _ in range(turn % 4):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(path), img)
    return str(path)


# --- D141: 3x4 + rhythm strips (FK and the like) ------------------------------------
# Four rows: three carry the lead groups of SEQ_GROUPS one after the other, 2.5 s each,
# without a gap between them (a dashed tick and the label mark the boundary); the
# fourth is lead II for the whole 10 s. Unlike the three-channel strips above, the
# groups and the rhythm row are simultaneous: one time axis for the photo.

RHYTHM_GROUP_S = 2.5          # each group's length (10 s / 4)
RHYTHM_WIN_S = 0.6            # windows the group kinds are judged in ...
RHYTHM_STEP_S = 0.1
RHYTHM_MAX_ERR = 0.3          # ... a window is limb / augmented when its identities hold to this residual
RHYTHM_CHEST_ERR = 0.8        # ... and a chest window when neither comes closer than this
RHYTHM_MIN_EVIDENCE_S = 0.6   # limb or augmented windows needed to name the leads (D138)
RHYTHM_MIN_WINDOW_S = 1.0     # a group part shorter than this on the photo is left out
RHYTHM_EDGE_S = 0.3           # trimmed off both ends of a group: a boundary found from one side
                              # alone was up to 0.4 s off on synthetic strips
RHYTHM_INK_RATIOS = (0.7, 0.78, 0.8, 0.62)
RHYTHM_MIN_ASPECT = 1.8       # paper this elongated may be such a strip (an A4 page: 1.41; a cut
                              # piece of the strip runs off the photo, so less than SEQ_MIN_ASPECT)
RHYTHM_MAX_FILL = 0.7         # paper filling more of the image is a page, not a strip on a desk
RHYTHM_MIN_PX_PER_MM = 8.0    # below this four rows are refused as too coarse (D141)
RHYTHM_LOW_RES = "rhythm strip: too coarse"


def _window_kinds(rows: dict[str, np.ndarray], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre (samples) and kind of every window: 0 = I-III (row 2 is the rhythm's II and
    I + III), 1 = aVR-aVF (they sum to zero, aVF - aVR = 1.5 II), 2 = chest (neither),
    -1 = undecided."""
    a, b, c, rh = rows["r0"], rows["r1"], rows["r2"], rows["r3"]
    n = min(len(a), len(b), len(c), len(rh))
    win, step = max(8, int(RHYTHM_WIN_S * fs)), max(2, int(RHYTHM_STEP_S * fs))
    centres, kinds = [], []
    def rel(x: np.ndarray, ref: np.ndarray) -> float:
        """Residual of x against ref, relative to ref's own spread."""
        x, ref = x - np.median(x), ref - np.median(ref)
        d = float(np.linalg.norm(ref))
        return float(np.linalg.norm(x - ref)) / d if d > 0 else np.inf

    for x in range(0, max(1, n - win + 1), step):
        s = slice(x, x + win)
        # Shape alone does not tell the groups apart: V5 often looks like II, and V5 tracks
        # V4 + V6. The identities hold in amplitude too: the limb group's II is the rhythm's
        # II and I + III; the augmented leads sum to zero and aVF - aVR = 1.5 II.
        limb_err = max(rel(b[s], rh[s]), rel(a[s] + c[s], b[s]))
        aug_err = max(rel(c[s] - a[s], 1.5 * rh[s]), rel(-(a[s] + c[s]), b[s]))
        if limb_err <= RHYTHM_MAX_ERR and limb_err <= aug_err:
            k = 0
        elif aug_err <= RHYTHM_MAX_ERR:
            k = 1
        elif min(limb_err, aug_err) >= RHYTHM_CHEST_ERR:
            k = 2
        else:
            k = -1
        centres.append(x + win // 2)
        kinds.append(k)
    return np.asarray(centres), np.asarray(kinds)


def _strip_start(centres: np.ndarray, kinds: np.ndarray, n: int, group_px: float) -> tuple[float, int]:
    """Column where the strip's time 0 falls (may lie left of the photo) and the evidence for
    it: limb and augmented windows in their own group, less every window out of place. A
    window in group k agrees when its kind is k (chest in groups 2-3)."""
    starts = np.arange(-4 * group_px, group_px, 2.0)
    fit, evidence = [], []
    for s0 in starts:
        g = np.floor((centres - s0) / group_px).astype(int)
        exp = np.where(g >= 2, 2, g)
        inside = (g >= 0) & (g <= 3)
        named = kinds >= 0
        agree = inside & named & (kinds == exp)
        # Chest windows count both ways too: they pin the end of the augmented group.
        against = named & ~agree & ((kinds <= 1) | (inside & (g <= 1)))
        fit.append(int(agree.sum()) - int(against.sum()))
        evidence.append(int(agree[kinds <= 1].sum()) - int(against.sum()))
    fit_a = np.asarray(fit)
    top = np.flatnonzero(fit_a == fit_a.max())
    # The middle of the best run of starts, not its first: one-sided evidence leaves a range.
    run = np.split(top, np.flatnonzero(np.diff(top) > 1) + 1)[0]
    i = int(run[len(run) // 2])
    return float(starts[i]), evidence[i]


def _low_res_text() -> str:
    try:
        from ecgscan.i18n import t
        return t("reject.LOW_RESOLUTION")
    except ImportError:
        return "photo too small: take it closer or send it as a file, uncompressed"


def is_rhythm_strip_candidate(bgr: np.ndarray) -> bool:
    found = find_paper(bgr)
    if found is None:
        return False
    h, w = found[0].shape[:2]
    # A strip lies on something: a scan or a render of a whole page fills the image, and the
    # page reader takes it (a 3x4 + rhythm page is this layout too).
    if h * w > RHYTHM_MAX_FILL * bgr.shape[0] * bgr.shape[1]:
        return False
    return max(h, w) >= RHYTHM_MIN_ASPECT * min(h, w)


def digitize_rhythm_strip(path: str | Path, *, mm_per_s: float = 25.0,
                          mm_per_mv: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    """``({lead: trace_uv}, {lead: (i0, i1)})`` for a photo of a 3x4 + rhythm strip: every
    trace on the photo's one time axis, valid inside its sample window; II from the rhythm
    row across the photo. ``digitize_rhythm_strip.last_calibration`` holds the provenance.
    Raises ValueError when four rows are not found or the leads cannot be named."""
    import cv2

    from cardiolens.layer1_ingestion.strip import digitize_strip

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    found = find_paper(bgr)
    if found is None:
        raise ValueError("3x4+rhythm strip: no gridded paper found")
    paper = found[0]
    turns = (0, 2) if paper.shape[1] >= paper.shape[0] else (1, 3)
    best = None
    for t, ink_ratio in ((t, r) for t in turns for r in RHYTHM_INK_RATIOS):
        view = np.ascontiguousarray(np.rot90(paper, -t)) if t else paper
        try:
            rows = digitize_strip(view, rows=4, lead_order=["r0", "r1", "r2", "r3"], mm_per_s=mm_per_s,
                                  mm_per_mv=mm_per_mv, ink_ratio=ink_ratio)
        except ValueError:
            continue
        calib = dict(digitize_strip.last_calibration)  # type: ignore[attr-defined]
        fs = float(calib["px_per_mm"] * mm_per_s)
        centres, kinds = _window_kinds(rows, fs)
        n = min(len(v) for v in rows.values())
        s0, score = _strip_start(centres, kinds, n, RHYTHM_GROUP_S * fs)
        if best is None or score > best[0]:
            best = (score, t, ink_ratio, rows, calib, fs, s0, n, kinds,
                    float(digitize_strip.last_coverage))  # type: ignore[attr-defined]
    if best is None:
        raise ValueError("3x4+rhythm strip: four rows not found")
    score, t, ink_ratio, rows, calib, fs, s0, n, kinds, cov = best
    win_s = RHYTHM_STEP_S
    if calib["px_per_mm"] < RHYTHM_MIN_PX_PER_MM:
        # D141: the FK strip at ~5 px/mm (Telegram) — four rows, but traces too coarse for the
        # lead identities to hold anywhere; the page reader gave 42 / 55 / 89 per piece. The
        # scanner's limit (8 px/mm, D139) holds here too, named or not.
        raise ValueError(f"{RHYTHM_LOW_RES} ({calib['px_per_mm']:.1f} px/mm): {_low_res_text()}")
    if score * win_s < RHYTHM_MIN_EVIDENCE_S:
        raise ValueError("3x4+rhythm strip: neither the I-III nor the aVR-aVF group is recognised on the photo "
                         "— the leads cannot be named; photograph the whole strip")
    group_px = RHYTHM_GROUP_S * fs
    edge = int(RHYTHM_EDGE_S * fs)
    traces: dict[str, np.ndarray] = {"II": rows["r3"][:n].astype(np.float32)}
    windows: dict[str, tuple[int, int]] = {"II": (0, n)}
    for k, leads in enumerate(SEQ_GROUPS):
        i0, i1 = int(max(0, s0 + k * group_px)) + edge, int(min(n, s0 + (k + 1) * group_px)) - edge
        if i1 - i0 < RHYTHM_MIN_WINDOW_S * fs:
            continue
        for r, ln in enumerate(leads):
            if ln == "II":
                continue
            tr = np.zeros(n, dtype=np.float32)
            seg = rows[f"r{r}"][i0:i1]
            tr[i0:i1] = seg - float(np.median(seg))
            traces[ln], windows[ln] = tr, (i0, i1)
    calib.update(layout="3x4+rhythm strip", rotated_deg=90 * t, ink_ratio=ink_ratio, fs=fs, mm_per_s=mm_per_s,
                 strip_start_px=round(s0, 1), match_windows=int(score),
                 leads=sorted(windows, key=STANDARD_12_LEADS.index))
    digitize_rhythm_strip.last_calibration = calib  # type: ignore[attr-defined]
    digitize_rhythm_strip.last_coverage = cov  # type: ignore[attr-defined]
    return traces, windows


def digitize_rhythm_strip_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    traces, windows = digitize_rhythm_strip(path, **kw)
    calib = digitize_rhythm_strip.last_calibration  # type: ignore[attr-defined]
    fs, n = calib["fs"], len(traces["II"])
    signal = np.zeros((12, n), dtype=np.float32)
    for ln, tr in traces.items():
        signal[STANDARD_12_LEADS.index(ln)] = tr
    note = (f"3x4+rhythm strip photo: leads {', '.join(calib['leads'])}, II from the rhythm row; "
            f"{calib['mm_per_s']} mm/s assumed, each group valid only inside its time window")
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS), duration_s=n / fs, source="image",
        provenance=Provenance(digitized=True, confidence=round(digitize_rhythm_strip.last_coverage, 3),  # type: ignore[attr-defined]
                              lead_windows={ln: (a / fs, b / fs) for ln, (a, b) in windows.items()},
                              notes=[f"calibration: {calib}", note]
                                    + ([SCALE_CONFLICT_NOTE] if calib.get("scale_conflict") else [])))


def render_rhythm_strip(sig: SignalRepresentation, path: str | Path, *, window_s: tuple[float, float] = (0.0, 10.0),
                        px_per_mm: int = 8, mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                        row_pitch_mm: float = 15.0, turn: int = 0) -> str:
    """A synthetic photo of a 3x4 + rhythm strip (tests): white gridded paper on a dark desk,
    four rows (the groups of SEQ_GROUPS, ``RHYTHM_GROUP_S`` each, back to back with a dashed
    tick and a label at each boundary; II for the whole record below), 1-mV pulses at the
    strip's start. ``window_s``: the part of the strip on the photo."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import _draw_trace

    fs = sig.sampling_rate_hz
    pps = mm_per_s * px_per_mm
    lead_in = int(10 * px_per_mm)          # the pulses before time 0
    t0, t1 = window_s
    full_w = lead_in + int(4 * RHYTHM_GROUP_S * pps) + int(5 * px_per_mm)
    pitch = int(row_pitch_mm * px_per_mm)
    h = pitch * 5
    paper = np.full((h, full_w, 3), (236, 236, 240), np.uint8)
    for x in range(0, full_w, px_per_mm):
        cv2.line(paper, (x, 0), (x, h), (215, 210, 225) if x % (5 * px_per_mm) else (185, 175, 210), 1)
    for y in range(0, h, px_per_mm):
        cv2.line(paper, (0, y), (full_w, y), (215, 210, 225) if y % (5 * px_per_mm) else (185, 175, 210), 1)
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    top = int(1000 * px_per_uv)
    for r in range(4):
        yc = (r + 1) * pitch
        pts = np.array([[1, 0], [3, 0], [3, -1], [8, -1], [8, 0], [9.5, 0]], np.float64)
        pts = np.column_stack([pts[:, 0] * px_per_mm, yc + pts[:, 1] * top]).astype(np.int32)
        cv2.polylines(paper, [pts.reshape(-1, 1, 2)], False, (30, 30, 30), 2)
    n = int(RHYTHM_GROUP_S * fs)
    for k, leads in enumerate(SEQ_GROUPS):
        x0 = lead_in + int(k * RHYTHM_GROUP_S * pps)
        for r, ln in enumerate(leads):
            yc = (r + 1) * pitch
            if k:
                for d in range(-3, 3, 2):
                    cv2.line(paper, (x0, yc + d * px_per_mm), (x0, yc + (d + 1) * px_per_mm), (40, 40, 40), 1)
            cv2.putText(paper, ln, (x0 + px_per_mm, yc - int(4 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.05 * px_per_mm, (40, 40, 40), 1, cv2.LINE_AA)
            _draw_trace(paper, sig.lead(ln)[k * n:(k + 1) * n], fs, x0, yc, pps, px_per_uv)
    _draw_trace(paper, sig.lead("II")[:4 * n], fs, lead_in, 4 * pitch, pps, px_per_uv)
    a = 0 if t0 <= 0 else lead_in + int(t0 * pps)
    b = min(full_w, lead_in + int(t1 * pps) + (int(5 * px_per_mm) if t1 >= 4 * RHYTHM_GROUP_S else 0))
    paper = paper[:, a:b]
    w = paper.shape[1]
    rng = np.random.default_rng(1)
    margin = int(0.25 * h)
    desk = rng.integers(20, 90, size=((h + 2 * margin) // 16 + 1, (w + 2 * margin) // 16 + 1, 3)).astype(np.uint8)
    img = cv2.resize(desk, ((w + 2 * margin) // 16 * 16 + 16, (h + 2 * margin) // 16 * 16 + 16),
                     interpolation=cv2.INTER_NEAREST)[:h + 2 * margin, :w + 2 * margin].copy()
    img[margin:margin + h, margin:margin + w] = paper
    for _ in range(turn % 4):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(path), img)
    return str(path)
