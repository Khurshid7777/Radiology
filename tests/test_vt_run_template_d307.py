"""D307: when a VT run is most of the strip, the run detector's template comes from the narrowest QRS (the conducted
beats), and the reference cycle is the one that ends on them (D306: 7 / 13 missed runs had the run as template)."""

import numpy as np

from cardiolens.arrhythmia import beats as B
from cardiolens.contracts.signal import Patient, SignalRepresentation

FS = 500


def _strip(pattern):
    """Leads II, V1, V5; 'N' = narrow upright QRS after a long cycle, 'V' = wide inverted QRS at ~140/min."""
    t, r = [], 300
    for kind in pattern:
        r += 500 if kind == "N" else 215          # samples: 1000 ms before a conducted beat, 430 ms (~140/min) in a run
        t.append((r, kind))
    n = r + 600
    x = np.zeros((3, n))
    tt = np.arange(n)
    for r, kind in t:
        if kind == "N":
            for k in range(3):
                x[k] += 1000 * np.exp(-0.5 * ((tt - r) / 6) ** 2)
        else:
            for k, sign in enumerate((-1, 1, -1)):
                x[k] += sign * 900 * np.exp(-0.5 * ((tt - r) / 22) ** 2)
    sig = SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=FS, lead_names=["II", "V1", "V5"],
                               duration_s=n / FS, source="synthetic", patient=Patient(age_years=60))
    return sig, [r for r, _ in t]


def test_run_that_dominates_the_strip_is_found():
    sig, rp = _strip("NVVVVNVVVVVN" "VVVVN")
    runs = B.detect_ventricular_runs(sig, rp, FS)
    assert runs and max(n for _, n, _ in runs) >= 4


def test_without_d307_the_same_strip_is_missed(monkeypatch):
    sig, rp = _strip("NVVVVNVVVVVN" "VVVVN")
    monkeypatch.setattr(B, "_RUN_TEMPLATE_NARROW_MS", None)
    assert not B.detect_ventricular_runs(sig, rp, FS)


def test_beat_widths_tell_narrow_from_wide():
    sig, rp = _strip("NVN")
    half = int(B._PVC_HALF_S * FS)
    segs = {n: np.stack([sig.lead(n)[r - half:r + half].astype(float) for r in rp]) for n in sig.lead_names}
    w = B._beat_widths(segs, sig.lead_names, FS)
    assert w[1] > w[0] + 20 and w[1] > w[2] + 20
