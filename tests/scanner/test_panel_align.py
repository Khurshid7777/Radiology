"""SD-168: the panels' windows put back in time by the rhythm strip's beats."""

from types import SimpleNamespace

import numpy as np

from ecgscan.contract.models import Trace
from ecgscan.layout.panels import Panel
from ecgscan.vectorize.align import align_readings, beat_period, fit_plane

FS = 500.0
CFG = {"envelope_s": 0.04, "min_strip_s": 8.0, "min_panels": 6, "min_peak": 0.3, "max_shift_s": 0.25, "rr_share": 0.45,
       "max_mad_s": 0.02, "min_span_s": 0.03}
LEADS = [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]


def ecg(seconds: float, rr: float, seed: int = 0) -> np.ndarray:
    """Narrow QRS spikes every ``rr`` s with a little jitter, a broad T wave and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = np.zeros_like(t)
    beat = 0.3
    while beat < seconds:
        x += 1.0 * np.exp(-((t - beat) / 0.012) ** 2) + 0.25 * np.exp(-((t - beat - 0.3) / 0.06) ** 2)
        beat += rr * (1 + rng.uniform(-0.04, 0.04))
    return x + rng.normal(0, 0.01, len(t))


def page(row_skew_s: float, col_skew_s: float = 0.0, rr: float = 0.8, strip: bool = True):
    """A 3x4+rhythm sheet read with each row ``row_skew_s`` later than the one above (a turned sheet)."""
    sig = ecg(10.0, rr)
    out = []
    for c, col in enumerate(LEADS):
        for r, lead in enumerate(col):
            true = c * 2.5
            read_start = true + r * row_skew_s + c * col_skew_s + 0.2      # the read places it this late
            a, b = int(true * FS), int((true + 2.5) * FS)
            amp = 0.3 if lead == "aVL" else 1.0                            # a small lead: its own match is poor
            p = Panel(lead=lead, kind="panel", row=r, t0_s=true, t1_s=true + 2.5, x0=int(c * 300), x1=int(c * 300 + 300),
                      baseline=100.0 + 120.0 * r)
            out.append(SimpleNamespace(panel=p, trace=Trace.from_array(amp * sig[a:b], start_s=read_start, fs=FS, lead=lead, quality=1.0)))
    if strip:
        p = Panel(lead="II", kind="rhythm", row=3, t0_s=0.0, t1_s=10.0, x0=0, x1=1200, baseline=460.0)
        out.append(SimpleNamespace(panel=p, trace=Trace.from_array(sig, start_s=0.2 + 3 * row_skew_s, fs=FS, lead="II", quality=1.0)))
    return out


def relative_starts(readings) -> dict[str, float]:
    """Each panel's start against the strip, minus where the truth puts it (0 = aligned)."""
    strip = next(r for r in readings if r.panel.kind == "rhythm").trace.start_s
    return {r.panel.lead: round(r.trace.start_s - strip - r.panel.t0_s, 3) for r in readings if r.panel.kind == "panel"}


def test_a_turned_sheet_is_put_back_in_time():
    rd = page(row_skew_s=0.03)
    before = relative_starts(rd)
    assert max(before.values()) - min(before.values()) > 0.05
    al = align_readings(rd, FS, CFG)
    after = relative_starts(rd)
    assert al.why_not is None and al.used >= 6
    assert max(abs(v) for v in after.values()) <= 0.012, after


def test_a_horizontal_scale_error_too():
    rd = page(row_skew_s=0.0, col_skew_s=-0.03)
    align_readings(rd, FS, CFG)
    assert max(abs(v) for v in relative_starts(rd).values()) <= 0.012


def test_a_square_sheet_and_a_sheet_without_a_strip_are_left_alone():
    rd = page(row_skew_s=0.005)
    starts = [r.trace.start_s for r in rd]
    al = align_readings(rd, FS, CFG)
    assert al.why_not == "small skew" and [r.trace.start_s for r in rd] == starts
    rd = page(row_skew_s=0.04, strip=False)
    starts = [r.trace.start_s for r in rd]
    assert align_readings(rd, FS, CFG).why_not == "no strip" and [r.trace.start_s for r in rd] == starts


def test_time_never_goes_below_zero():
    rd = page(row_skew_s=-0.05)                 # rows earlier and earlier: the first column would start before 0
    al = align_readings(rd, FS, CFG)
    assert al.why_not is None and al.offset_s >= 0
    assert min(r.trace.start_s for r in rd) >= 0
    assert max(abs(v) for v in relative_starts(rd).values()) <= 0.012


def test_a_fast_heart_limits_the_search_to_less_than_half_a_beat():
    assert abs(beat_period(ecg(10.0, 0.4), FS, win_s=0.04) - 0.4) < 0.03
    assert abs(beat_period(ecg(10.0, 1.0), FS, win_s=0.04) - 1.0) < 0.05
    assert beat_period(np.random.default_rng(1).normal(0, 1, 5000), FS, win_s=0.04) is None


def test_the_plane_drops_a_match_far_off_it():
    pts = np.array([[y, x] for y in (100, 220, 340) for x in (150, 450, 750, 1050)], float)
    shifts = 0.0002 * pts[:, 0] - 0.0001 * pts[:, 1]
    shifts[4] += 0.2                            # one odd lead
    coef, mad, keep = fit_plane(pts, shifts, np.ones(len(shifts)), min_n=6)
    assert not keep[4] and keep.sum() == 11 and mad < 1e-6
    assert np.allclose(coef[:2], [0.0002, -0.0001], atol=1e-6)
