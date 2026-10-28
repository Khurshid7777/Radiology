"""A study on several sheets (TZ 18.4, S2): one record printed or cut over 2..N pages.

Each sheet is read on its own (``ScanHints.piece``: a piece shorter than its layout is not
refused for it). Speed and gain are those of the first sheet, which carries the pulse and the
header; the later sheets are read with them. A lead on two neighbouring sheets is one record
cut in two: joined where the next sheet repeats the last seconds of the previous one. The
overlap is found by the beats, never assumed: the next sheet's first R wave is put on each R
wave of the previous sheet's tail in turn, and the shift whose RR intervals and curve both
repeat is taken. A regular rhythm repeats itself one beat later as well; when two shifts fit
alike the seam is not guessed. No overlap, an ambiguous one, or an RR interval across the
seam out of the rhythm around it: ``SEAM_UNCERTAIN`` (the doctor checks the order and the
seam). Leads on one sheet only (12 leads printed on two pages) are kept as read.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from ecgscan.contract import Layout, Page, ScanResult, Trace

MIN_OVERLAP_S = 0.5        # a shorter repeat is not told from chance
OVERLAP_ALIKE = 0.6        # the repeated seconds this alike (1 - rms(diff) / rms, smoothed) are the same seconds
OVERLAP_SMOOTH_S = 0.008   # ... smoothed this much: two reads of one curve differ by a sample here and there
SEAM_RR_TOL = 0.15         # the RR across a seam this far from the median RR around it: a break
EDGE_TRIM_S = 0.1          # a sheet cut through the trace: its last / first 0.1 s run into the paper edge
MIN_OVERLAP_BEATS = 3      # two RR intervals at least in the overlap: one RR fits a beat-shifted match too often
RR_MATCH_S = 0.010         # the overlapping beats' RR intervals agree to this (mean |difference|) ...
RR_DISTINCT_S = 0.008      # ... and better by this than any other shift, else the overlap is ambiguous


@dataclasses.dataclass
class Seam:
    lead: str
    overlap_s: float | None    # None: none found, joined end to end
    at_s: float                # where the next sheet starts in the joined record
    rr_ratio: float | None     # RR across the seam / median RR around it


def _finite_span(x: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(np.isfinite(x))
    return None if idx.size == 0 else (int(idx[0]), int(idx[-1]) + 1)


def _alike(a: np.ndarray, b: np.ndarray, sigma: float = 0.0) -> float:
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        a, b = gaussian_filter1d(np.nan_to_num(a, nan=np.nanmedian(a)), sigma), \
            gaussian_filter1d(np.nan_to_num(b, nan=np.nanmedian(b)), sigma)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return -1.0
    a, b = a[ok] - np.median(a[ok]), b[ok] - np.median(b[ok])
    ra = float(np.sqrt(np.mean(a ** 2)))
    return -1.0 if ra == 0 else 1.0 - float(np.sqrt(np.mean((a - b) ** 2))) / ra


def find_overlap(a: np.ndarray, b: np.ndarray, fs: float, *, min_s: float = MIN_OVERLAP_S,
                 need: float = OVERLAP_ALIKE) -> tuple[int | None, bool]:
    """(samples by which ``b`` starts before ``a`` ends, ambiguous). None: no overlap found;
    ambiguous: two shifts fit as well (a regular rhythm), so none is taken."""
    pa, pb = r_peaks(a, fs), r_peaks(b, fs)
    if len(pb) < 2 or len(pa) < 2:
        return None, False
    fits = []
    for t in pa:
        shift = int(t - pb[0])                  # b's sample 0 sits at a's sample ``shift``
        lap = len(a) - shift
        if lap < min_s * fs or shift <= 0 or lap > 0.9 * len(b):
            continue
        a_beats = pa[pa >= t]
        b_beats = pb[pb + shift < len(a)] + shift
        k = min(len(a_beats), len(b_beats))
        if k < MIN_OVERLAP_BEATS:
            continue
        rr_err = float(np.mean(np.abs(np.diff(a_beats[:k]) - np.diff(b_beats[:k])))) / fs
        best = max(((_alike(a[shift + d:], b[:lap - d], OVERLAP_SMOOTH_S * fs), lap - d)
                    for d in range(-int(0.02 * fs), int(0.02 * fs) + 1)
                    if 0 < lap - d <= len(b) and shift + d >= 0), default=(-1.0, lap))
        fits.append((rr_err, best[0], best[1]))
    good = sorted(f for f in fits if f[1] >= need and f[0] <= RR_MATCH_S)
    if not good:
        return None, False
    if len(good) > 1 and good[1][0] - good[0][0] < RR_DISTINCT_S:
        return None, True
    return good[0][2], False


def r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """R peaks of one lead: the largest deflection, at least 0.3 s apart."""
    from scipy.signal import find_peaks

    y = np.nan_to_num(x - np.nanmedian(x))
    if np.max(np.abs(y)) == 0:
        return np.array([], dtype=int)
    # The 99th percentile, not the maximum: one spike (a fold, the paper's edge) does not hide
    # every beat under its half height.
    hi, lo = np.percentile(y, 99), -np.percentile(y, 1)
    if lo > hi:
        y, hi = -y, lo
    peaks, _ = find_peaks(y, height=0.5 * hi, distance=int(0.3 * fs))
    return peaks


def seam_rr(x: np.ndarray, at: int, fs: float) -> float | None:
    """RR across sample ``at`` over the median RR of the beats around it (None: too few beats)."""
    p = r_peaks(x, fs)
    if len(p) < 4:
        return None
    k = int(np.searchsorted(p, at))
    if k == 0 or k >= len(p):
        return None
    rr = np.diff(p)
    across = rr[k - 1]
    around = np.delete(rr, k - 1)
    return float(across / np.median(around)) if around.size else None


def join(a: np.ndarray, b: np.ndarray, fs: float) -> tuple[np.ndarray, int | None, int]:
    """(joined, overlap in samples or None, index where b starts)."""
    sa, sb = _finite_span(a), _finite_span(b)
    if sa is None:
        return b, None, 0
    if sb is None:
        return a, None, len(a)
    edge = int(EDGE_TRIM_S * fs)
    a, b = a[sa[0]:max(sa[0] + 1, sa[1] - edge)], b[min(sb[1] - 1, sb[0] + edge):sb[1]]
    lap, _ = find_overlap(a, b, fs)
    if lap is None:
        # End to end: each sheet's own baseline level (a sheet without the pulse may sit off).
        return np.concatenate([a, b - np.nanmedian(b) + np.nanmedian(a)]), None, len(a)
    b = b + np.nanmedian(a[-lap:] - b[:lap])          # the next sheet on this one's level over the overlap
    return np.concatenate([a[:-lap], b]), lap, len(a) - lap


def stitch(pages: list[ScanResult], shas: list[str], fs: float) -> tuple[ScanResult, list[Seam]]:
    """The study from its sheets read one by one (all read at ``fs``), and its seams."""
    first = pages[0]
    if any(p.status == "rejected" for p in pages):
        reasons = sorted({r for p in pages for r in p.reject_reasons})
        return first.model_copy(update={"status": "rejected", "reject_reasons": reasons, "leads": {}, "rhythm": [],
                                        "pages": _pages(pages, shas, {})}), []
    leads: dict[str, np.ndarray] = {}
    seams: list[Seam] = []
    overlaps: dict[int, list[float | None]] = {}
    for i, page in enumerate(pages):
        for name, tr in page.leads.items():
            x = tr.array()
            if name not in leads:
                leads[name] = x
                continue
            joined, lap, at = join(leads[name], x, fs)
            ratio = seam_rr(joined, at, fs)
            seams.append(Seam(name, None if lap is None else lap / fs, at / fs, ratio))
            overlaps.setdefault(i - 1, []).append(None if lap is None else lap / fs)
            leads[name] = joined
    warnings = [w for w in dict.fromkeys(w for p in pages for w in p.warnings)]
    if seams:
        warnings.append("PAGES_STITCHED")
    if any(s.overlap_s is None or (s.rr_ratio is not None and abs(s.rr_ratio - 1) > SEAM_RR_TOL) for s in seams):
        warnings.append("SEAM_UNCERTAIN")
    status = "needs_confirmation" if (any(p.status != "ok" for p in pages) or "SEAM_UNCERTAIN" in warnings) else "ok"
    names = {p.layout.name for p in pages}
    layout = Layout(name=first.layout.name if len(names) == 1 else None,
                    confidence=min(p.layout.confidence for p in pages), registry_version=first.layout.registry_version,
                    confirmed_by_user=first.layout.confirmed_by_user)
    out_leads = {name: Trace.from_array(x, start_s=0.0, fs=fs, lead=name,
                                        quality=min(p.leads[name].quality for p in pages if name in p.leads))
                 for name, x in leads.items()}
    joined_sha = hashlib.sha256("".join(shas).encode()).hexdigest()
    source = first.source.model_copy(update={"original_sha256": joined_sha, "parts_sha256": shas})
    res = first.model_copy(update={
        "status": status, "warnings": warnings, "leads": out_leads, "rhythm": [r for p in pages for r in p.rhythm],
        "layout": layout, "source": source, "pages": _pages(pages, shas, overlaps)})
    return ScanResult.model_validate(res.model_dump()), seams


def _pages(pages: list[ScanResult], shas: list[str], overlaps: dict[int, list[float | None]]) -> list[Page]:
    out = []
    for i, (p, sha) in enumerate(zip(pages, shas, strict=True)):
        laps = [v for v in overlaps.get(i, []) if v is not None]
        last = i == len(pages) - 1
        out.append(Page(index=i, sha256=sha, status=p.status, layout=p.layout.name,
                        overlap_with_next=None if last or i not in overlaps else bool(laps),
                        overlap_s=round(float(np.median(laps)), 3) if laps else None))
    return out


def scan_pages(images: list, *, hints=None, fs_out: float = 500.0, **kw) -> tuple[ScanResult, list[Seam]]:
    """Read every sheet, the first one's speed and gain passed to the rest; stitch them."""
    from ecgscan.pipeline import ScanHints, scan

    hints = hints or ScanHints()
    data = [im if isinstance(im, bytes) else Path(im).read_bytes() for im in images]
    shas = [hashlib.sha256(d).hexdigest() for d in data]
    first = scan(data[0], hints=dataclasses.replace(hints, piece=True), fs_out=fs_out, **kw)
    cal = first.calibration
    later = dataclasses.replace(hints, piece=True,
                                paper_speed_mm_s=hints.paper_speed_mm_s or cal.paper_speed_mm_s,
                                gain_mm_mv=hints.gain_mm_mv or cal.gain_mm_mv)
    pages = [first] + [scan(d, hints=later, fs_out=fs_out, **kw) for d in data[1:]]
    return stitch(pages, shas, fs_out)
