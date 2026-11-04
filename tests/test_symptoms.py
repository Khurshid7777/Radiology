"""D195: the ticked symptoms add advice and urgency, never take a finding or an alert away,
and never reach the narrative's input."""

import pytest

import cardiolens.pipeline as pl
from cardiolens.api.service import signal_from_case
from cardiolens.clinical_context import advise, clean
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import SCALE_CONFLICT_NOTE
from cardiolens.outputs.conclusion import format_conclusion


def _run(case, symptoms=(), onset=None, mode="A", sig=None):
    sig = sig or signal_from_case(case)
    sig.context.symptoms, sig.context.symptom_onset = clean(list(symptoms), onset)
    return pl.run_pipeline(sig, mode=mode)


def _same(a, b):
    return (a.findings.codes() == b.findings.codes()
            and [x.code for x in a.safety.alerts] == [x.code for x in b.safety.alerts]
            and a.measurements.heart_rate_bpm == b.measurements.heart_rate_bpm)


def test_without_symptoms_nothing_changes():
    a, b = _run("stemi"), _run("stemi")
    assert _same(a, b) and a.symptom_advice == [] and a.report.mode == "A"
    assert "Симптомы" not in format_conclusion(a, "ru")


@pytest.mark.parametrize("case,symptoms", [("stemi", ["chest_pain"]), ("normal", ["chest_pain"]),
                                           ("brady", ["syncope"]), ("normal", ["syncope"]),
                                           ("normal", ["palpitations"]), ("normal", ["dyspnea"])])
def test_symptoms_never_take_away_a_finding_or_an_alert(case, symptoms):
    base, with_s = _run(case), _run(case, symptoms)
    assert base.findings.codes() <= with_s.findings.codes()
    assert {x.code for x in base.safety.alerts} <= {x.code for x in with_s.safety.alerts}


def test_chest_pain_with_st_elevation_is_acute_coronary_and_early_is_said():
    r = _run("stemi", ["chest_pain"], "lt12h")
    a = r.symptom_advice[0]
    assert a.urgent and "остром коронарном" in a.ru and "12 ч" in a.ru and "koronar" in a.uz


def test_chest_pain_with_a_normal_record_asks_to_repeat_the_ecg():
    a = _run("normal", ["chest_pain"]).symptom_advice[0]
    assert "не исключает инфаркт" in a.ru and "15–30" in a.ru and a.uz


def test_chest_pain_on_a_gated_photo_says_do_not_wait():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    sig.provenance.notes = [SCALE_CONFLICT_NOTE]
    r = _run(None, ["chest_pain"], sig=sig)
    assert "photo_unreliable" in r.findings.codes()
    a = r.symptom_advice[0]
    assert "не ждите" in a.ru and "не исключает" not in a.ru


def test_syncope_rows():
    rhythm = advise(["syncope"], None, {"long_qt"}, {"long_qt": "Удлинение QT"})[0]
    plain = advise(["syncope"], None, {"sinus_rhythm"}, {})[0]
    assert "Удлинение QT" in rhythm.ru and "аритмическая причина" in rhythm.ru
    assert "не исключает аритмию" in plain.ru and plain.ru != rhythm.ru


def test_palpitations_without_an_arrhythmia_point_to_a_holter():
    assert "холтер" in advise(["palpitations"], None, {"sinus_rhythm"}, {})[0].ru
    assert advise(["palpitations"], None, {"atrial_fibrillation"}, {}) == []


def test_every_row_has_its_own_text_in_both_languages():
    rows = [advise(["chest_pain"], None, {"st_elevation"}, {}), advise(["chest_pain"], None, set(), {}),
            advise(["chest_pain"], None, {"photo_unreliable"}, {}), advise(["syncope"], None, {"long_qt"}, {}),
            advise(["syncope"], None, set(), {}), advise(["palpitations"], None, set(), {})]
    texts = [r[0] for r in rows]
    assert len({t.ru for t in texts}) == len(texts) and len({t.uz for t in texts}) == len(texts)
    assert all(t.ru and t.uz for t in texts)


def test_mode_b_goes_to_a_doctor_with_chest_pain_or_syncope_only():
    assert _run("normal", ["chest_pain"], mode="B").report.mode == "escalate"
    assert _run("normal", ["syncope"], mode="B").report.mode == "escalate"
    assert _run("normal", ["palpitations"], mode="B").report.mode == _run("normal", mode="B").report.mode


def test_symptoms_never_reach_the_narrative(monkeypatch):
    seen = {}
    real = pl.interpret

    def spy(data, *a, **k):
        seen["ctx"] = data.context
        return real(data, *a, **k)
    monkeypatch.setattr(pl, "interpret", spy)
    _run("stemi", ["chest_pain"], "lt12h")
    assert seen["ctx"].symptoms == [] and seen["ctx"].symptom_onset is None


def test_unknown_codes_are_dropped():
    assert clean(["chest_pain", "<script>", "syncope"], "soon") == (["chest_pain", "syncope"], None)


def test_the_conclusion_and_so_the_pdf_carry_the_symptoms_and_advice():
    text = format_conclusion(_run("stemi", ["chest_pain"], "lt12h"), "ru")
    assert "Симптомы: боль в груди (менее 12 ч назад)" in text and "остром коронарном" in text
