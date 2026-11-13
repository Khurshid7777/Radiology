"""D265: an SVT with a bundle-branch-block pattern is a wide-complex tachycardia (rule out VT); a VT run cut
by the strip's edge counts when it is long and clearly faster than the rhythm."""
import numpy as np

import cardiolens.layer4_findings.rules as R
from cardiolens.arrhythmia import beats as B
from cardiolens.config import norms_for_patient
from cardiolens.contracts.findings import Finding
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure


def _svt_findings(monkeypatch, with_bbb: bool):
    sig = generate_synthetic_ecg(heart_rate_bpm=170)
    m = measure(sig)
    real = R._conduction_findings

    def fake(m_, norms, fs, patient=None):
        real(m_, norms, fs, patient)
        if with_bbb and fs.by_code("lbbb") is None:
            fs.findings.append(Finding(code="lbbb", label_ru="БЛНПГ", label_uz="LBBB", confidence=0.8))

    monkeypatch.setattr(R, "_conduction_findings", fake)
    return R.detect_findings(sig, m, norms=norms_for_patient(40, None)).codes()


def test_svt_with_bbb_pattern_is_wide_complex_tachycardia(monkeypatch):
    codes = _svt_findings(monkeypatch, with_bbb=True)
    assert "wide_complex_tachycardia" in codes and "supraventricular_tachycardia" not in codes


def test_narrow_svt_unchanged(monkeypatch):
    codes = _svt_findings(monkeypatch, with_bbb=False)
    assert "wide_complex_tachycardia" not in codes


def test_edge_run_parameters():
    assert B._RUN_EDGE_MIN_BEATS == 4 and 0 < B._RUN_EDGE_FASTER < B._RUN_FASTER_THAN_REF


def test_edge_run_arithmetic():
    """Eight beats at 600 ms, then five abnormal ones at 360 ms reaching the strip's end: long enough,
    >= 100/min and clearly faster than the reference RR, so the edge rule accepts the run."""
    fs = 500.0
    rp = np.cumsum([300] + [300] * 8 + [180] * 5)
    rr = np.diff(rp).astype(float)
    med = float(np.percentile(rr, B._RUN_TEMPLATE_RR_PCT))
    j, e = 9, 14
    assert e - j >= B._RUN_EDGE_MIN_BEATS
    assert float(np.mean(rr[j:e - 1] / fs * 1000.0)) <= B._RUN_MAX_RR_MS
    assert float(np.mean(rr[j:e - 1])) <= B._RUN_EDGE_FASTER * med
