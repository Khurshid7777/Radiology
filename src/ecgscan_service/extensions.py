"""P2 extension points (TZ 18.11, SD-097): consultation, a patient's archive and serial
comparison, QR, a doctor's signature. Only the seams are here: the interfaces a later stage
implements, and switches that are off.

``configs/scanner/service.yaml`` → ``extensions``: one switch per feature. ``features()`` is what
the app asks for (``GET /scans/config/features``) to decide what to show; today all are off.
Switching one on without registering its implementation refuses to start: a feature is either
whole or absent, never half there.

What each implementation must keep (from TZ 18.9, 18.12):

* **ShareLinks** — a link for a colleague: random token, expiry, revocable, one scan only; the
  opening is written to the scan's audit. No sharing without the doctor's explicit action.
* **Comments** — text on a place of the signal (``LeadRef``: an export channel and a time span),
  stored with the scan, removed with it; no names in the text (the same limit as confirmations).
* **Notifier** — tells a colleague a scan was shared or commented; carries the scan id only.
* **SubjectArchive** — a patient's scans by ``ScanResult.subject_ref`` (an issued pseudonym,
  never a name); ``previous_analysis`` gives the core its ``prior_ecg_id`` for serial comparison.
* **Signer** — signs ``ScanResult.content_sha256()`` of a confirmed reading for the doctor;
  the key never leaves the signer.
"""

from __future__ import annotations

from typing import Protocol

from ecgscan.config import load_yaml
from ecgscan.contract.models import LeadRef, Signature

FEATURES = ("consult", "archive", "qr", "signature")


class ShareLinks(Protocol):
    def create(self, scan_id: str, *, by: str, ttl_h: float) -> str: ...
    def revoke(self, token: str, *, by: str) -> None: ...
    def resolve(self, token: str) -> str | None: ...          # scan id while valid


class Comments(Protocol):
    def add(self, scan_id: str, *, by: str, text: str, at: LeadRef | None = None) -> str: ...
    def list(self, scan_id: str) -> list[dict]: ...


class Notifier(Protocol):
    def notify(self, recipient: str, event: str, scan_id: str) -> None: ...


class SubjectArchive(Protocol):
    def scans(self, subject_ref: str, *, owner: str) -> list[str]: ...
    def previous_analysis(self, subject_ref: str, *, before_scan: str, owner: str) -> str | None: ...


class Signer(Protocol):
    def sign(self, content_sha256: str, *, by: str) -> Signature: ...


# What each switch needs registered before it may be on.
REQUIRES: dict[str, tuple[str, ...]] = {
    "consult": ("share_links", "comments", "notifier"),
    "archive": ("subject_archive",),
    "qr": ("subject_archive",),          # a patient's code binds a scan to their archive
    "signature": ("signer",),
}

_registry: dict[str, object] = {}


def register(name: str, impl: object) -> None:
    """Install an implementation (``share_links``, ``comments``, ``notifier``,
    ``subject_archive``, ``signer``)."""
    known = {n for names in REQUIRES.values() for n in names}
    if name not in known:
        raise ValueError(f"unknown extension {name!r}; expected one of {sorted(known)}")
    _registry[name] = impl


def get(name: str) -> object | None:
    return _registry.get(name)


def switches() -> dict[str, bool]:
    conf = load_yaml("service").get("extensions") or {}
    unknown = set(conf) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown extensions in service.yaml: {sorted(unknown)}")
    return {f: bool(conf.get(f, False)) for f in FEATURES}


def features() -> dict[str, bool]:
    """Which P2 features are on. One switched on without its implementation is an error."""
    on = switches()
    for feature, enabled in on.items():
        missing = [n for n in REQUIRES[feature] if n not in _registry]
        if enabled and missing:
            raise RuntimeError(f"extension {feature!r} is on but {missing} not registered")
    return on
