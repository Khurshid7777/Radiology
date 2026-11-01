"""D189: a photo gated only for its limb leads keeps the rate read off a good lead-II strip;
axis, voltages and intervals stay hidden."""

import cardiolens.pipeline as pl
from cardiolens.contracts.findings import released_rate
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.strip import GRID_IN_TRACE_NOTE, LEADS_INCONSISTENT_NOTE
from cardiolens.outputs.pdf_report import measurement_line


def _strip(monkeypatch, agree=True, tc=0.99):
    real = pl.measure

    def m(*a, **k):
        r = real(*a, **k)
        r.page_beat_agreement = agree
        r.qrs_template_consistency = tc
        return r
    monkeypatch.setattr(pl, "measure", m)


def _scan(*extra, warnings="LIMB_LEADS_INCONSISTENT,TRACE_GAPS"):
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    sig.provenance.notes = ["ecgscan scan_id=x status=needs_confirmation", f"warnings: {warnings}",
                            LEADS_INCONSISTENT_NOTE, *extra]
    return sig


def test_a_limb_only_gate_keeps_the_rate_and_hides_the_rest(monkeypatch):
    _strip(monkeypatch)
    res = pl.run_pipeline(_scan(), mode="A")
    rate = released_rate(res.findings)
    assert rate is not None and abs(rate - 72) <= 3
    assert res.findings.codes() == {"photo_unreliable"}       # no axis, voltage or interval finding
    f = res.findings.by_code("photo_unreliable")
    assert f"ЧСС {rate:.0f} /мин по полосе II" in f.label_ru and "ось, вольтаж" in f.label_ru
    line = measurement_line(res)
    assert line.startswith(f"ЧСС: {rate:.0f} /мин") and "QRS" not in line and "Ось, вольтаж" in line
    assert not res.safety.alerts


def test_the_scanners_rhythm_doubt_keeps_the_rate_hidden(monkeypatch):
    """1480 v1, 1510 v2, 1629 v2: tc 0.99-1.0 and beats on the columns, rate 7 /min high."""
    _strip(monkeypatch)
    res = pl.run_pipeline(_scan(warnings="LIMB_LEADS_INCONSISTENT,RHYTHM_INCONSISTENT"), mode="A")
    assert "photo_unreliable" in res.findings.codes() and released_rate(res.findings) is None
    assert measurement_line(res).startswith("Измерения не приводятся")


def test_a_weak_strip_keeps_the_rate_hidden(monkeypatch):
    for agree, tc in ((False, 0.99), (True, 0.6), (None, 0.99)):
        _strip(monkeypatch, agree=agree, tc=tc)
        assert released_rate(pl.run_pipeline(_scan(), mode="A").findings) is None


def test_another_gate_keeps_the_rate_hidden(monkeypatch):
    _strip(monkeypatch)
    assert released_rate(pl.run_pipeline(_scan(GRID_IN_TRACE_NOTE), mode="A").findings) is None


def test_a_readers_page_keeps_the_rate_hidden(monkeypatch):
    _strip(monkeypatch)
    sig = _scan()
    sig.provenance.notes = sig.provenance.notes[2:]            # no scanner note: the readers read it
    assert released_rate(pl.run_pipeline(sig, mode="A").findings) is None


def test_the_narrative_gets_the_rate_and_nothing_else(monkeypatch):
    from cardiolens.layer6_interpret.claude import _measurements
    from cardiolens.layer6_interpret.provider import InterpretationInput
    _strip(monkeypatch)
    res = pl.run_pipeline(_scan(), mode="A")
    d = _measurements(InterpretationInput(findings=res.findings, causal_graph=res.causal_graph,
                                          measurements=res.measurements, patient=res.signal.patient,
                                          context=res.signal.context, mode="A", critical_labels=[],
                                          disclaimer_ru="", disclaimer_uz=""))
    assert set(d) == {"heart_rate_bpm", "withheld"} and d["heart_rate_bpm"] == released_rate(res.findings)
