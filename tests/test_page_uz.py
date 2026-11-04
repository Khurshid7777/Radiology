"""D200: every Russian phrase of the page's markup has an Uzbek line, and the switch is there."""

import re
from html.parser import HTMLParser
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "src/cardiolens/api/static/index.html").read_text(encoding="utf-8")
# The administrator's quality block stays Russian for now (D200).
SKIP = {"Качество за 30 дней (по журналу, без данных пациентов)"}


def _phrases() -> list[str]:
    body = PAGE[PAGE.index("<body"):PAGE.index("<script>")]
    out: list[str] = []

    class P(HTMLParser):
        def handle_data(self, d):
            t = " ".join(d.split())
            if re.search("[А-Яа-яЁё]", t):
                out.append(t)

        def handle_starttag(self, tag, attrs):
            out.extend(" ".join(v.split()) for k, v in attrs
                       if k in ("placeholder", "title", "alt") and v and re.search("[А-Яа-яЁё]", v))
    P().feed(body)
    return [p for p in dict.fromkeys(out) if p not in SKIP]


def _uz_keys() -> set[str]:
    block = PAGE[PAGE.index("const UZ = {"):PAGE.index("};", PAGE.index("const UZ = {"))]
    return {k.replace("\\'", "'") for k in re.findall(r"'((?:[^'\\]|\\.)+)'\s*:", block)}


def test_every_phrase_of_the_markup_has_an_uzbek_line():
    missing = [p for p in _phrases() if p not in _uz_keys()]
    assert not missing, missing


def test_the_uzbek_lines_are_latin():
    block = PAGE[PAGE.index("const UZ = {"):PAGE.index("};", PAGE.index("const UZ = {"))]
    values = re.findall(r":\s*\n?\s*'((?:[^'\\]|\\.)+)'", block)
    assert values and not [v for v in values if re.search("[А-Яа-яЁё]", v)]


def test_the_switch_is_on_the_page():
    assert 'id="lang_ru"' in PAGE and 'id="lang_uz"' in PAGE and "function applyLang()" in PAGE
