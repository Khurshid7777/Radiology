"""D193/D194: the page offers the prior-ECG comparison and turns a portrait photo before upload."""

from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "src/cardiolens/api/static/index.html").read_text(encoding="utf-8")


def test_the_page_sends_the_prior_and_shows_the_comparison():
    assert "fd.append('prior_ecg_id'" in PAGE and 'id="serial_box"' in PAGE and "renderSerial(d.serial_comparison)" in PAGE


def test_the_browser_keeps_only_number_and_time():
    # D210 (owner): no rate in the browser either — only the analysis number and the time.
    line = next(x for x in PAGE.splitlines() if "{ id: String(d.analysis_id)" in x)
    assert "t: Date.now() }" in line and "hr" not in line and "conclusion" not in line and "age" not in line


def test_a_portrait_photo_waits_for_the_user_to_turn_it():
    assert "if (rot.pending)" in PAGE and 'id="rotl"' in PAGE and 'id="rotr"' in PAGE
    assert "c.width = bmp.height; c.height = bmp.width" in PAGE     # full size, sides swapped
