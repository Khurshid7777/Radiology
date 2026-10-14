"""Delta-wave descriptor and the delta route of the WPW rule (D51)."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.pipeline import run_pipeline


def _sig():
    return SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
                                duration_s=0.02, source="synthetic", patient=Patient(age_years=40, sex="male"))


def _find(m):
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_delta_route_needs_slurred_upstroke_and_a_short_enough_pr():
    base = dict(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=False)
    slurred = Measurements(**base, morphology=Morphology(delta_ms=60.0),
                           intervals=Intervals(pr_ms=130, qrs_ms=104, qt_ms=380, qtc_bazett_ms=420))
    assert "wpw_preexcitation" in _find(slurred)
    sharp = slurred.model_copy(update={"morphology": Morphology(delta_ms=22.0)})
    assert "wpw_preexcitation" not in _find(sharp)
    long_pr = slurred.model_copy(update={"intervals": Intervals(pr_ms=170, qrs_ms=104, qt_ms=380, qtc_bazett_ms=420)})
    assert "wpw_preexcitation" not in _find(long_pr)
    # RSR' in V1 with a wide QRS is RBBB, whatever the initial slope.
    rbbb = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=True,
                        morphology=Morphology(delta_ms=60.0),
                        intervals=Intervals(pr_ms=130, qrs_ms=140, qt_ms=380, qtc_bazett_ms=420))
    codes = _find(rbbb)
    assert "rbbb" in codes and "wpw_preexcitation" not in codes


def test_classic_route_still_fires():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True,
                     intervals=Intervals(pr_ms=95, qrs_ms=130, qt_ms=380, qtc_bazett_ms=420))
    assert "wpw_preexcitation" in _find(m)


def test_delta_descriptor_short_on_a_normal_qrs_and_long_on_a_slurred_one():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    m = run_pipeline(sig, mode="A").measurements
    assert m.morphology.delta_ms is not None and m.morphology.delta_ms < 40
    assert "wpw_preexcitation" not in run_pipeline(sig, mode="A").findings.codes()
    # Slur every QRS onset: replace R-60 ms .. R-10 ms (onset and upstroke; the QRS is
    # 70 ms) by a slow ramp into the complex (a delta wave), in every lead.
    res = run_pipeline(sig, mode="A")
    fs = sig.sampling_rate_hz
    slurred = sig.model_copy(deep=True)
    for r in res.measurements.r_peaks_samples:
        on, off = r - int(0.06 * fs), r - int(0.01 * fs)
        if on < 0:
            continue
        for li in range(slurred.signal.shape[0]):
            a, b = slurred.signal[li, on], slurred.signal[li, off]
            slurred.signal[li, on:off] = np.linspace(a, b, off - on)
    m2 = run_pipeline(slurred, mode="A").measurements
    assert m2.morphology.delta_ms is not None and m2.morphology.delta_ms > m.morphology.delta_ms
