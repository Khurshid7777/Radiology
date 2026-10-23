"""Does the wave before each QRS repeat from beat to beat? (D147)

A P wave — sinus, ectopic, or in sinus arrhythmia — has nearly the same shape before
every beat; fibrillatory waves do not. The window 300-60 ms before each R in lead II
(band 1-30 Hz) is compared with the median of all windows:

* ``corr`` — median correlation of each window with that template (irregular rhythm
  without AF on PTB-XL: median 0.94; AF: 0.44-0.51);
* ``keep`` — template amplitude over the mean window amplitude: an organised wave
  survives the averaging, fibrillation averages out.

The P-wave detector takes coarse f-waves or noise for "P" on many AF records; this does
not ask whether there is a wave, only whether it is the same wave each time.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation


def p_window_consistency(sig: SignalRepresentation, r_peaks: list[int], *, lead: str = "II",
                         window_ms: tuple[float, float] = (300.0, 60.0),
                         min_beats: int = 4) -> tuple[float, float] | None:
    """``(corr, keep)`` for ``lead``, or ``None`` when the lead or the beats are missing."""
    if lead not in sig.lead_names or len(r_peaks) < min_beats:
        return None
    from scipy.signal import butter, filtfilt

    fs = float(sig.sampling_rate_hz)
    x = np.asarray(sig.signal[sig.lead_names.index(lead)], dtype=np.float64)
    if not np.isfinite(x).all() or len(x) < int(fs):
        return None
    b, a = butter(2, [1.0 / (fs / 2), min(30.0, 0.45 * fs) / (fs / 2)], "band")
    s = filtfilt(b, a, x)
    w0, w1 = int(window_ms[0] / 1000 * fs), int(window_ms[1] / 1000 * fs)
    segs = np.array([s[r - w0:r - w1] for r in r_peaks if r - w0 >= 0 and r - w1 <= len(s)])
    if len(segs) < min_beats:
        return None
    segs = segs - segs.mean(axis=1, keepdims=True)
    tmpl = np.median(segs, axis=0)
    if tmpl.std() == 0:
        return None
    cors = [float(np.corrcoef(sg, tmpl)[0, 1]) for sg in segs if sg.std() > 0]
    if len(cors) < min_beats:
        return None
    keep = float(tmpl.std() / (np.mean(segs.std(axis=1)) + 1e-9))
    return float(np.median(cors)), keep
