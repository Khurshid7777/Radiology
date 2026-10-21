"""RU/UZ strings of the scanner, the API and the bot (SD-016).

One source: ``configs/scanner/i18n/{ru,uz}.json`` (key -> text, ``{name}``
placeholders). Uzbek is in Latin script. A missing key is an error, never a
silent fallback to the key or to the other language.
"""

from __future__ import annotations

import json
from functools import cache

from ecgscan.config import config_dir

LANGS = ("ru", "uz")


@cache
def strings(lang: str) -> dict[str, str]:
    if lang not in LANGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGS}")
    return json.loads((config_dir() / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def t(key: str, lang: str = "ru", **fmt) -> str:
    return strings(lang)[key].format(**fmt)
