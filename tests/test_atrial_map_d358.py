"""D358 / D361: the same conducted P before most QRS withdraws a false AF (on since D361)."""

import numpy as np
import pytest

import cardiolens.pipeline as P
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer3_measure.atrial_map import LEADS, conducted_p


@pytest.fixture(scope="module")
def sinus():
    sig = generate_synthetic_ecg(heart_rate_bpm=70, seed=3)
    res = P.run_pipeline(sig, mode="A")
    return sig, res.measurements


def _leads(sig):
    return {ln: np.asarray(sig.lead(ln), float) for ln in LEADS if ln in sig.lead_names}


def test_sinus_has_the_same_p_before_every_qrs(sinus):
    sig, m = sinus
    f = conducted_p(_leads(sig), m.r_peaks_samples, sig.sampling_rate_hz)
    if f is None or f["n_p"] == 0:
        pytest.skip("P detector model not available")
    assert f["share"] >= 0.9 and f["pr_iqr_s"] <= 0.02 and f["shape_ii"] >= 0.9


def test_too_few_beats_gives_nothing(sinus):
    sig, m = sinus
    assert conducted_p(_leads(sig), m.r_peaks_samples[:3], sig.sampling_rate_hz) is None


def _af_set():
    return FindingSet(findings=[Finding(code="atrial_fibrillation", group="rhythm", label_ru="ФП", label_uz="BF", confidence=0.8)])


def test_off_keeps_af(sinus, monkeypatch):
    sig, m = sinus
    assert P.AF_SAME_P_WITHDRAW is True        # D361: on (owner accepted)
    monkeypatch.setattr(P, "AF_SAME_P_WITHDRAW", False)
    fs, audit = _af_set(), []
    P._withdraw_af_on_same_p(sig, m, fs, audit)
    assert "atrial_fibrillation" in fs.codes() and not audit


def test_on_withdraws_af_and_says_why(sinus, monkeypatch):
    sig, m = sinus
    f = conducted_p(_leads(sig), m.r_peaks_samples, sig.sampling_rate_hz)
    if f is None or f["n_p"] == 0:
        pytest.skip("P detector model not available")
    monkeypatch.setattr(P, "AF_SAME_P_WITHDRAW", True)
    fs, audit, cache = _af_set(), [], {}
    assert P._withdraw_af_on_same_p(sig, m, fs, audit, cache) is True and "f" in cache
    assert "atrial_fibrillation" not in fs.codes()
    irr = fs.by_code("irregular_rhythm")
    assert irr is not None and "ФП не подтверждается" in irr.label_ru and "D358" in audit[-1]
    assert not any("а" <= ch <= "я" for ch in irr.label_uz.lower())


def test_visible_atrial_activity_keeps_af(sinus, monkeypatch):
    """D359: f-waves or an atrial (flutter) rate keep the AF whatever the P before the QRS look like."""
    sig, m = sinus
    monkeypatch.setattr(P, "AF_SAME_P_WITHDRAW", True)
    for field, value in (("atrial_rate_bpm", 280.0), ("fwave_index", 1.2)):
        mm = m.model_copy(update={field: value})
        fs, audit = _af_set(), []
        P._withdraw_af_on_same_p(sig, mm, fs, audit)
        assert "atrial_fibrillation" in fs.codes() and not audit


def test_qt_is_assessed_as_in_a_regular_rhythm_after_the_af_goes():
    """D365: "QTc not assessable in AF" becomes a long QT with the same QTc; "approximate in AF" goes."""
    from cardiolens.contracts.findings import Evidence

    fs = FindingSet(findings=[
        Finding(code="qtc_unreliable_af", group="interval", label_ru="QTc при фибрилляции предсердий не оценивается",
                label_uz="BF da QTc baholanmaydi", confidence=0.5,
                evidence=[Evidence(feature="qtc_ms", value="478"), Evidence(feature="formula", value="Bazett"),
                          Evidence(feature="af_long_min", value="500")])])
    audit = []
    P._qt_after_af_withdrawn(fs, audit)
    qt = fs.by_code("long_qt")
    assert "qtc_unreliable_af" not in fs.codes() and qt is not None and "478" in qt.label_ru and "ФП" not in qt.label_ru
    assert "D365" in audit[-1] and not qt.critical
    fs2 = FindingSet(findings=[Finding(code="long_qt", group="interval", confidence=0.85,
                                       label_ru="Удлинение QT (QTc 520 мс, Bazett, при ФП ориентировочно)",
                                       label_uz="QT uzayishi (QTc 520 ms, Bazett, BF da taxminiy)")])
    P._qt_after_af_withdrawn(fs2, [])
    assert fs2.by_code("long_qt").label_ru.endswith("Bazett)") and "BF" not in fs2.by_code("long_qt").label_uz
