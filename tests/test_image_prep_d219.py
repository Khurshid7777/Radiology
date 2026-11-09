"""D219: small pictures are enlarged before reading; a long QT read off them near the limit is approximate."""

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.layer1_ingestion import image_prep
from cardiolens.layer1_ingestion.image_prep import ENLARGED_NOTE, prepared


def test_small_picture_is_enlarged_and_a_large_one_is_not(tmp_path: Path):
    small, large = tmp_path / "s.jpg", tmp_path / "l.jpg"
    Image.new("RGB", (1000, 700), "white").save(small)
    Image.new("RGB", (3000, 2000), "white").save(large)
    with prepared(small) as p:
        assert Path(p) != small and Image.open(p).size == (2000, 1400)
    with prepared(large) as p:
        assert Path(p) == large
    with prepared(small, {**image_prep.PREP, "enabled": False}) as p:
        assert Path(p) == small


def _long_qt(qtc: float) -> FindingSet:
    return FindingSet(findings=[Finding(code="long_qt", group="interval", label_ru="Удлинение QT", label_uz="QT", confidence=0.85,
                                        evidence=[Evidence(feature="qtc_ms", value=str(qtc)),
                                                  Evidence(feature="upper", value="450")])])


_SIG = SimpleNamespace(provenance=SimpleNamespace(notes=[ENLARGED_NOTE]))


def test_long_qt_near_the_limit_from_an_enlarged_picture_is_approximate():
    from cardiolens.pipeline import _approximate_long_qt

    fs = _long_qt(470)
    _approximate_long_qt(_SIG, fs, [])
    assert fs.codes() == {"long_qt_approximate", "photo_low_resolution"}
    fs = _long_qt(495)
    _approximate_long_qt(_SIG, fs, [])
    assert fs.codes() == {"long_qt", "photo_low_resolution"}
