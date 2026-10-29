"""SD-089: a speed that OCR returns as two overlapping boxes ('25mm/', 'n/s') is read; a bare
'25mm/' or bare numbers are not."""

import pytest

from ecgscan.calibration.header_ocr import parse_settings

KIT_12390 = ["ID: 123901hr", "Name:", "25", "10", "10r", "ID: 12390_hr", "Name:", "25mn", "25mm/", "n/s",
             "10mm/mV", "10mm/mV"]     # the colour read, then the brightest-channel read (SD-088)


def test_the_kit_footer_gives_speed_and_gain():
    assert parse_settings(KIT_12390) == ([25.0], [10.0, 10.0])


@pytest.mark.parametrize("texts, speeds", [
    (["25mm/", "m/s"], [25.0]),
    (["50 мм/", "м/с"], [50.0]),
    (["25mm/"], []),                      # no tail: not a speed
    (["25mm/", "10mm/mV"], []),           # the next box is the gain, not the tail
    (["Age: 25", "10"], []),              # bare numbers are not settings
])
def test_split_speed(texts, speeds):
    assert parse_settings(texts)[0] == speeds
