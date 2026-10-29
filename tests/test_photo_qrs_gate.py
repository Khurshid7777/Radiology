"""D186: a digitised QRS narrower than any adult's gates the photo (ecgdig v5/v6 photos)."""

import cardiolens.pipeline as pl
from cardiolens.layer1_ingestion import generate_synthetic_ecg


def _with_qrs(monkeypatch, qrs_ms):
    real = pl.measure

    def narrow(*a, **k):
        m = real(*a, **k)
        m.intervals.qrs_ms = qrs_ms
        return m
    monkeypatch.setattr(pl, "measure", narrow)


def test_a_photo_read_with_a_12_ms_qrs_is_gated(monkeypatch):
    _with_qrs(monkeypatch, 12.0)
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    res = pl.run_pipeline(sig, mode="A")
    f = next(f for f in res.findings.findings if f.code == "photo_unreliable")
    assert "QRS 12 мс" in f.label_ru and "QRS 12 ms" in f.label_uz


def test_the_same_qrs_in_a_digital_file_is_not_a_photo_gate(monkeypatch):
    _with_qrs(monkeypatch, 12.0)
    res = pl.run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    assert "photo_unreliable" not in res.findings.codes()


def test_a_narrow_but_possible_qrs_photo_passes(monkeypatch):
    _with_qrs(monkeypatch, 62.0)
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.provenance.digitized = True
    assert "photo_unreliable" not in pl.run_pipeline(sig, mode="A").findings.codes()
