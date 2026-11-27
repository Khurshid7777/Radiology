"""D364: AF with wide QRS at >= 100/min keeps the AF and adds the urgent 'rule out VT' note (no red alert)."""

import cardiolens.pipeline as P
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.findings import Finding, FindingSet

ADULT = {"age_group": "adult"}


def _m(hr, qrs):
    return Measurements(heart_rate_bpm=hr, intervals=Intervals(qrs_ms=qrs))


def _af(extra=()):
    fs = [Finding(code="atrial_fibrillation", group="rhythm", label_ru="ФП", label_uz="BF", confidence=0.8)]
    fs += [Finding(code=c, group="rhythm", label_ru=c, label_uz=c, confidence=0.8) for c in extra]
    return FindingSet(findings=fs)


def test_wide_fast_af_gets_the_note_and_keeps_af():
    fs, audit = _af(), []
    P._af_wide_note(_m(130, 140), ADULT, fs, audit)
    note = fs.by_code("wide_qrs_tachycardia")
    assert note is not None and "atrial_fibrillation" in fs.codes() and not note.critical
    assert "исключите ЖТ" in note.label_ru and "D364" in audit[-1]
    assert not any("а" <= ch <= "я" for ch in note.label_uz.lower())


def test_no_note_when_narrow_slow_pediatric_or_vt_named():
    for m, norms, extra in ((_m(130, 110), ADULT, ()), (_m(90, 140), ADULT, ()),
                            (_m(130, 140), {"age_group": "pediatric"}, ()),
                            (_m(150, 140), ADULT, ("ventricular_tachycardia",))):
        fs, audit = _af(extra), []
        P._af_wide_note(m, norms, fs, audit)
        assert "wide_qrs_tachycardia" not in fs.codes() and not audit


def test_no_note_without_af():
    fs, audit = FindingSet(findings=[]), []
    P._af_wide_note(_m(130, 140), ADULT, fs, audit)
    assert not fs.findings


def test_note_goes_with_the_af_on_a_page_without_a_rhythm_strip():
    """D259 drops AF on a 3x4 page with no rhythm strip; D364's note mentions AF, so it goes too."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg

    fs, audit = _af(), []
    P._af_wide_note(_m(130, 140), ADULT, fs, audit)
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.lead_windows = {"II": (0.0, 2.5), "V1": (2.5, 5.0)}
    P._withhold_rhythm_without_strip(sig, fs, audit)
    assert "atrial_fibrillation" not in fs.codes() and "wide_qrs_tachycardia" not in fs.codes()
