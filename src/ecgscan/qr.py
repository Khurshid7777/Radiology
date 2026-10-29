"""QR payloads (P2 extension point, TZ 18.11, SD-097): the format is fixed now so that the
clinic's printed codes and a later app agree. Nothing reads QR codes yet.

    cv1:s/<subject_ref>    bind the scan to a patient (``ScanResult.subject_ref``)
    cv1:l/<token>          sign in to the web version: a one-time token, 32-128 URL-safe chars

The prefix carries a version, so a future format is a new prefix, never a reinterpretation.
Anything else, a URL included, is refused: a code must not send the phone to an arbitrary site
or carry a name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ecgscan.contract.models import SUBJECT_REF_PATTERN

PREFIX = "cv1:"
_SUBJECT = re.compile(SUBJECT_REF_PATTERN)
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


@dataclass(frozen=True)
class QrPayload:
    kind: Literal["subject", "login"]
    value: str


def parse_qr(text: str) -> QrPayload:
    """The payload of a scanned code; ValueError for anything not in the format above."""
    if not text.startswith(PREFIX):
        raise ValueError("not a CardioVision code")
    body = text[len(PREFIX):]
    kind, _, value = body.partition("/")
    if kind == "s" and _SUBJECT.match(value):
        return QrPayload("subject", value)
    if kind == "l" and _TOKEN.match(value):
        return QrPayload("login", value)
    raise ValueError("malformed CardioVision code")


def subject_qr(subject_ref: str) -> str:
    """The text of a patient's code (the clinic's system prints it)."""
    if not _SUBJECT.match(subject_ref):
        raise ValueError("subject_ref must be an issued pseudonym (sub_…)")
    return f"{PREFIX}s/{subject_ref}"
