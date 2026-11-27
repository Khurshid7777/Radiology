"""D305: AF at a slow ventricular rate keeps its code and asks to rule out AV block II-III (Chapman: 10 / 19 blocks
named AF had HR < 55)."""

from cardiolens.config import load_norms
from cardiolens.layer4_findings import detect_findings
from test_af_fwaves import _m, _sig


def _af(hr, age="adult"):
    m = _m(0.2, 0.7, 0.6)
    m.heart_rate_bpm = hr
    fs = detect_findings(_sig(), m, norms=load_norms(age), la_ra_reversed=False)
    return next(f for f in fs.findings if f.code == "atrial_fibrillation")


def test_slow_af_keeps_the_code_and_gets_the_av_block_note():
    f = _af(48)
    assert "исключите АВ-блокаду" in f.label_ru and "AV blokadani" in f.label_uz
    # the note ends the text, after the slow-response qualifier (review of 18.11)
    assert f.label_ru.endswith("(проверьте зубцы P)") and f.label_uz.endswith("(P tishlarini tekshiring)")
    assert "медленн" in f.label_ru.lower()


def test_af_at_55_and_above_has_no_note():
    for hr in (55, 90):
        assert "АВ-блокаду" not in _af(hr).label_ru


def test_children_have_no_note():
    assert "АВ-блокаду" not in _af(48, "pediatric").label_ru
