"""R5 (SD-105): on a phone photo the page-wide check of the grid's vertical lines misses them;
rows found many and a multiple of 5 mm apart are the grid itself and start its removal."""

from ecgscan.layout.detect import LayoutGuess
from ecgscan.pipeline import _rows_are_grid

CFG = {"rows_min": 10, "rows_pitch_tol": 0.1}


def _guess(rows, pitch):
    return LayoutGuess(name=None, confidence=0.0, rows=rows, pitch_mm=pitch, baselines=[], uncertain=True)


def test_many_rows_on_the_5_mm_lattice_are_the_grid():
    assert _rows_are_grid(_guess(21, 10.15), CFG)          # ecgdig 1098-v4
    assert _rows_are_grid(_guess(12, 5.2), CFG)


def test_a_page_of_real_rows_is_not_the_grid():
    assert not _rows_are_grid(_guess(4, 35.0), CFG)        # 3x4 + rhythm
    assert not _rows_are_grid(_guess(12, 12.4), CFG)       # off the lattice
    assert not _rows_are_grid(_guess(6, 10.0), CFG)        # too few
    assert not _rows_are_grid(_guess(21, 10.0), {})        # off by config
