"""D254: ventricular couplets — two unlike-shaped beats in a row (outside a run of >= 3)."""
import numpy as np

from cardiolens.arrhythmia import couplet_finding, detect_ventricular_couplets
from cardiolens.contracts.signal import Patient, SignalRepresentation

FS = 500


def _strip(pvc_at: list[int]):
    """A 10-s two-lead strip at 75/min with narrow beats; beats listed in ``pvc_at`` are wide, inverted and early."""
    n = 10 * FS
    x = np.zeros((2, n))
    t = np.arange(-60, 61)
    narrow = 1000 * np.exp(-(t / 8.0) ** 2)
    wide = -900 * np.exp(-(t / 30.0) ** 2)
    rp, r = [], 300
    k = 0
    while r < n - 200:
        shape = wide if k in pvc_at else narrow
        x[:, r - 60:r + 61] += shape
        rp.append(r)
        r += 300 if (k + 1) in pvc_at else 400
        k += 1
    sig = SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=FS, lead_names=["II", "V1"],
                               duration_s=10.0, source="synthetic", patient=Patient(age_years=60, sex="male"))
    return sig, rp


def test_a_couplet_is_found():
    sig, rp = _strip([8, 9])
    assert len(detect_ventricular_couplets(sig, rp, FS)) == 1


def test_isolated_pvcs_and_triplets_are_not_couplets():
    sig, rp = _strip([5, 14])
    assert detect_ventricular_couplets(sig, rp, FS) == []
    sig, rp = _strip([8, 9, 10])
    assert detect_ventricular_couplets(sig, rp, FS) == []


def test_finding_is_not_critical():
    f = couplet_finding([1000])
    assert f.code == "ventricular_couplets" and not f.critical
