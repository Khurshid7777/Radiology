"""D221: a low anterior T against its QRS adds an 'old aneurysm?' note to the STEMI alert; the alert stays."""

from cardiolens.contracts.features import Measurements, Morphology, TWave
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer4_findings.rules import _aneurysm_note


def _fs():
    return FindingSet(findings=[Finding(code="st_elevation", group="ischemia", label_ru="Элевация ST",
                                        label_uz="ST", confidence=0.8)])


def _m(ratios):
    return Measurements(morphology=Morphology(t_waves=[TWave(lead=ld, amplitude_uv=100, qrs_ratio=r)
                                                       for ld, r in zip(("V1", "V2", "V3", "V4"), ratios)]))


def test_low_t_over_qrs_adds_the_note_and_keeps_the_alert():
    fs = _fs()
    _aneurysm_note(_m([0.2, 0.3, 0.25, 0.1]), ["V2", "V3"], fs)
    assert fs.codes() == {"st_elevation"} and "аневризм" in fs.by_code("st_elevation").label_ru


def test_one_tall_t_means_no_note():
    fs = _fs()
    _aneurysm_note(_m([0.2, 0.6, 0.25, 0.1]), ["V2", "V3"], fs)
    assert "аневризм" not in fs.by_code("st_elevation").label_ru


def test_inferior_elevation_gets_no_note():
    fs = _fs()
    _aneurysm_note(_m([0.2, 0.3, 0.25, 0.1]), ["II", "III", "aVF"], fs)
    assert "аневризм" not in fs.by_code("st_elevation").label_ru
