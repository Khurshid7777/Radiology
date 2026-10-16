"""Layout from the trace rows of a rectified page, checked against the registry (TZ §5.6).

Two independent cues:

* the number of trace rows equals a registry layout's rows (+ rhythm strips);
* the median baseline pitch (mm) matches a layout's ``row_pitch_mm``.

Both -> high confidence; the count alone -> medium (printers vary the pitch, e.g. 38 mm);
the pitch alone -> low confidence and a ``LAYOUT_UNCERTAIN`` warning (a stray
row from a shadow or a fold changes the count, rarely the median pitch); neither ->
unknown. Lead labels (OCR) join in M3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.layout.registry import available_layouts
from ecgscan.layout.rows import row_baselines


@dataclass
class LayoutGuess:
    name: str | None
    confidence: float
    rows: int
    pitch_mm: float | None
    baselines: list[int]
    uncertain: bool


def detect_layout(ink: np.ndarray, px_per_mm: float, *, pitch_tol: float = 0.1) -> LayoutGuess:
    base = row_baselines(ink, px_per_mm)
    n = len(base)
    pitch = float(np.median(np.diff(base))) / px_per_mm if n > 1 else None
    by_count, by_pitch = [], []
    for spec in available_layouts().values():
        if spec.n_trace_rows == n:
            by_count.append(spec.name)
        if pitch is not None and abs(pitch - spec.row_pitch_mm) <= pitch_tol * spec.row_pitch_mm:
            by_pitch.append(spec.name)
    both = [x for x in by_count if x in by_pitch]
    if len(both) == 1:
        return LayoutGuess(both[0], 0.9, n, pitch, base, False)
    if len(by_count) == 1:
        return LayoutGuess(by_count[0], 0.7, n, pitch, base, False)
    if len(by_pitch) == 1:
        return LayoutGuess(by_pitch[0], 0.4, n, pitch, base, True)
    return LayoutGuess(None, 0.0, n, pitch, base, True)
