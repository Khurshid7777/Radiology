"""D210 (UI brief, phases 1-2): what the page must keep — the safety rules of the brief, checked on the
markup and script (the browser check is tools/ui/check_p2.js)."""

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "src/cardiolens/api/static/index.html").read_text(encoding="utf-8")


def _between(a: str, b: str) -> str:
    return PAGE[PAGE.index(a):PAGE.index(b)]


def test_no_invented_progress():
    # The server reports no stages: no timed "reading… measuring… writing…" script.
    assert "STAGES" not in PAGE and "Читаю ЭКГ со снимка" not in PAGE
    assert 'id="progress" role="status" aria-live="polite"' in PAGE and 'id="err" role="alert"' in PAGE


def test_warnings_live_in_the_first_layer():
    first = _between('<div id="lay1"', '<div id="lay2"')
    for el in ('id="banner"', 'id="critical"', 'id="warnbox"', 'id="overlay_box"', 'id="disc"',
               "Если линии не на кривых, заключению верить нельзя"):
        assert el in first, el


def test_the_mode_stays_visible_on_the_form():
    form = _between('id="card2"', 'id="more"')
    assert 'id="mode_a"' in form and 'id="mode_b"' in form


def test_key_values_are_the_api_values_unformatted():
    body = PAGE[PAGE.index("function renderKeys"):PAGE.index("// D210 P3: a scale only")]
    assert "m.heart_rate_bpm" in body and "toFixed" not in body and "Math.round" not in body
    count = PAGE[PAGE.index("function countUp"):PAGE.index("// D210 P3: the ECG viewer")]
    assert "p.e.textContent = p.s" in count and "D = 600" in count        # the last frame is the API's text


def test_scales_only_from_the_server_ranges():
    body = PAGE[PAGE.index("function scale(v, r, u)"):PAGE.index("function countUp")]
    assert "r[0] == null || r[1] == null" in body and "n[nk]" in PAGE and "d.norms" in PAGE


def test_motion_respects_reduced_motion_and_moves_only_transform_or_opacity():
    assert "prefers-reduced-motion: reduce" in PAGE
    frames = re.findall(r"@keyframes \w+ \{(.*?)\}\s*\}", PAGE, flags=re.DOTALL)
    assert frames and all(not re.search(r"\b(width|height|top|left|margin)\s*:", f) for f in frames)


def test_rescan_keeps_the_patient_visibly_and_new_patient_clears():
    assert 'id="rescan"' in PAGE and 'id="newpat"' in PAGE and 'id="keepnote"' in PAGE
    newpat = PAGE[PAGE.index("$('newpat').addEventListener"):PAGE.index("// D196")]
    assert "'age'" in newpat and "symptom" in newpat and "sex = ''" in newpat
