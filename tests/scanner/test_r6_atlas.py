"""R6 (SD-106): atlas pages — a pulse stands at a row's edge, the grid scale agrees with the lines'
own period, a "row" crowded against another is a deep complex or a label line."""

from types import SimpleNamespace

import numpy as np

from ecgscan.geometry.grid import scale_matches_lines
from ecgscan.layout.detect import drop_crowded_rows
from ecgscan.pipeline import pulses_at_row_edges


def _pulse(x0, x1):
    return SimpleNamespace(x_left=x0, x_right=x1, top=10, bottom=40)


def test_a_pulse_mid_row_is_a_complex_not_a_pulse():
    ink = np.zeros((50, 1000), bool)
    ink[30, 100:900] = True
    kept = pulses_at_row_edges(ink, [_pulse(105, 125), _pulse(480, 520), _pulse(870, 895)], 0.15)
    assert [p.x_left for p in kept] == [105, 870]


def test_the_scale_must_match_the_lines_period():
    img = np.full((200, 2000, 3), 245, np.uint8)
    img[:, ::8] = (180, 180, 240)              # 1-mm lines at 8.16 px/mm, every 5th bolder
    for x in range(0, 2000, 41):
        img[:, x] = (120, 120, 220)
    assert scale_matches_lines(img, 8.16) is True
    assert scale_matches_lines(img, 16.3) is False       # the 2x error read on atlas Hampton


def test_crowded_rows_lose_the_weaker_and_even_rows_stay():
    ppm = 10.0
    ink = np.zeros((1600, 400), bool)
    for y in (370, 890, 1420):                   # three real rows, 52 mm apart
        ink[y - 3:y + 3, :] = True
    ink[1300:1303, 150:170] = True              # a deep S wave's tip 12 mm off: little ink
    assert drop_crowded_rows(ink, [370, 890, 1300, 1420], ppm, 0.3) == [370, 890, 1420]
    even = [300, 620, 940, 1260]
    assert drop_crowded_rows(ink, even, ppm, 0.3) == even
