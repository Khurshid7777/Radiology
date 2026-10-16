"""Bridge to the CardioLens trace extraction (SD-003): used until M3 replaces it.

M2 owns geometry and calibration; the traces are still read by
``cardiolens.layer1_ingestion`` (page tracker D47/D52, strip reader D72) on the
page ``ecgscan`` has already rectified and calibrated, so the legacy code no
longer guesses px/mm, mm/s or mm/mV. ``cardiolens`` is imported lazily here only.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import numpy as np

from ecgscan.contract import Trace


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    return np.interp(np.arange(n_out) / fs_out, np.arange(len(x)) / fs_in, x)


def parse_calibration(notes: list[str]) -> dict:
    for n in notes:
        if n.startswith("calibration: "):
            try:
                return ast.literal_eval(n[len("calibration: "):])
            except (ValueError, SyntaxError):
                return {}
    return {}


def count_rows(gray: np.ndarray) -> int:
    import cv2

    from cardiolens.layer1_ingestion.strip import count_rows as _count

    with tempfile.TemporaryDirectory(prefix="ecgscan_") as d:
        p = Path(d) / "page.png"
        cv2.imwrite(str(p), gray)
        return _count(p)


def signal_traces(sig, *, fs_out: float, quality: float) -> tuple[dict[str, Trace], list[Trace]]:
    """Legacy ``SignalRepresentation`` -> (panel traces, rhythm strips) at ``fs_out``, mV.

    A window markedly longer than the others (the 10-s lead II under a 3x4 page) is a
    rhythm strip.
    """
    fs = float(sig.sampling_rate_hz)
    windows = sig.provenance.lead_windows or {n: (0.0, sig.duration_s) for n in sig.lead_names}
    lengths = np.array([t1 - t0 for t0, t1 in windows.values()])
    typical = float(np.median(lengths)) if lengths.size else 0.0
    leads: dict[str, Trace] = {}
    rhythm: list[Trace] = []
    for lead, (t0, t1) in windows.items():
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        mv = resample_to(sig.lead(lead)[i0:i1].astype(np.float64) / 1000.0, fs, fs_out)
        if not len(mv):
            continue
        tr = Trace.from_array(mv, start_s=t0, fs=fs_out, lead=lead, quality=round(quality, 3), method="dp_path")
        (rhythm.append(tr) if t1 - t0 > 1.5 * typical else leads.__setitem__(lead, tr))
    return leads, rhythm


def read_traces(gray: np.ndarray, *, layout: str, px_per_mm: float, speed_mm_s: float, gain_mm_mv: float):
    """Run the legacy reader on a rectified, calibrated page; returns its SignalRepresentation."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import digitize_page, page_to_signal
    from cardiolens.layer1_ingestion.strip import digitize_strip, strip_to_signal

    with tempfile.TemporaryDirectory(prefix="ecgscan_") as d:
        p = Path(d) / "page.png"
        cv2.imwrite(str(p), gray)
        if layout == "12x1":
            traces = digitize_strip(p, rows=12, px_per_mm=px_per_mm, mm_per_s=speed_mm_s, mm_per_mv=gain_mm_mv,
                                    dewarp=False)
            return strip_to_signal(traces, px_per_mm=px_per_mm, mm_per_s=speed_mm_s,
                                   coverage=getattr(digitize_strip, "last_coverage", None),
                                   calibration=getattr(digitize_strip, "last_calibration", None))
        traces = digitize_page(p, px_per_mm=px_per_mm, mm_per_s=speed_mm_s, mm_per_mv=gain_mm_mv, dewarp=False)
        return page_to_signal(traces, px_per_mm=px_per_mm, mm_per_s=speed_mm_s, mm_per_mv=gain_mm_mv,
                              coverage=getattr(digitize_page, "last_coverage", None),
                              calibration=getattr(digitize_page, "last_calibration", None))
