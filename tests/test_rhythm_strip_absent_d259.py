"""D259: a digitised page with no lead >= 5 s (plain 3x4, no rhythm strip) names no rhythm from
RR irregularity; rate, morphology and a critical pause stay."""

from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.pipeline import RHYTHM_MIN_S, _withhold_rhythm_without_strip

_COLS = (("I", "II", "III"), ("aVR", "aVL", "aVF"), ("V1", "V2", "V3"), ("V4", "V5", "V6"))


def _page(strip_s: float | None):
    sig = generate_synthetic_ecg()
    windows = {ln: (2.5 * c, 2.5 * (c + 1)) for c, col in enumerate(_COLS) for ln in col}
    if strip_s is not None:
        windows["II"] = (0.0, strip_s)
    prov = sig.provenance.model_copy(update={"digitized": True, "lead_windows": windows})
    return sig.model_copy(update={"provenance": prov})


def _set(*codes, critical=()):
    return FindingSet(findings=[Finding(code=c, label_ru=c, label_uz=c, confidence=0.8,
                                        critical=c in critical) for c in codes])


def test_no_strip_withholds_irregularity_findings_only():
    fs = _set("atrial_fibrillation", "irregular_rhythm", "ventricular_ectopy", "sinus_pause",
              "rbbb", "sinus_tachycardia", "right_axis_deviation")
    audit: list[str] = []
    _withhold_rhythm_without_strip(_page(None), fs, audit)
    assert fs.codes() == {"rbbb", "sinus_tachycardia", "right_axis_deviation"}
    assert audit and "D259" in audit[0]


def test_critical_pause_stays():
    fs = _set("sinus_pause", "atrial_fibrillation", critical=("sinus_pause",))
    _withhold_rhythm_without_strip(_page(None), fs, [])
    assert fs.codes() == {"sinus_pause"}


def test_rhythm_strip_keeps_everything():
    fs = _set("atrial_fibrillation", "irregular_rhythm")
    _withhold_rhythm_without_strip(_page(10.0), fs, [])
    assert fs.codes() == {"atrial_fibrillation", "irregular_rhythm"}
    fs = _set("atrial_fibrillation")
    _withhold_rhythm_without_strip(_page(RHYTHM_MIN_S), fs, [])   # a 5-s strip (6x2) is a strip
    assert fs.codes() == {"atrial_fibrillation"}


def test_device_record_untouched():
    fs = _set("atrial_fibrillation", "irregular_rhythm")
    _withhold_rhythm_without_strip(generate_synthetic_ecg(), fs, [])
    assert fs.codes() == {"atrial_fibrillation", "irregular_rhythm"}
