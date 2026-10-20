"""DICOM C-STORE receiver for ECG devices (D96).

An ECG device configured to "send to PACS" (a DICOM storage SCU) pushes each recording
here; each 12-lead / general / ambulatory ECG object is handed to an ``analyze``
callable (by default a POST to the CardioLens API ``/api/analyze``) and the device gets
success or a failure status back. Verification (C-ECHO) is answered so the device's
"test connection" works.

Only calling AE titles on the whitelist are accepted, and with an empty whitelist the
listener does not start. Originals stay in memory; the JSONL log holds the time, the
device's AE title, a SHA-256 of the SOP Instance UID, the outcome and the analysis id —
no names, IDs or dates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from cardiolens.config import _load_yaml, _repo_root, configs_dir

log = logging.getLogger(__name__)

ECG_STORAGE = ("TwelveLeadECGWaveformStorage", "GeneralECGWaveformStorage",
               "General32bitECGWaveformStorage", "AmbulatoryECGWaveformStorage")
STATUS_OK = 0x0000
STATUS_OUT_OF_RESOURCES = 0xA700      # too large
STATUS_CANNOT_UNDERSTAND = 0xC000     # not an ECG we can read, or the analysis refused it


class ListenerConfig(BaseModel):
    ae_title: str = "CARDIOLENS"
    host: str = "127.0.0.1"
    port: int = 11112
    allowed_calling_aes: list[str] = Field(default_factory=list)
    max_bytes: int = 25 * 1024 * 1024
    api_url: str = "http://127.0.0.1:8000"
    api_timeout_s: float = 60.0
    log_path: str = "data/dicom_inbox/received.jsonl"
    # D98. An AE title is a plaintext name any client can claim: on a network the
    # source address is checked too (empty = any), and the receive is capped.
    allowed_ips: list[str] = Field(default_factory=list)
    max_associations: int = 4
    max_pdu_bytes: int = 65536
    api_key_env: str = "CARDIOLENS_LISTENER_API_KEY"   # the X-API-Key it sends to /api/analyze


def load_listener_config(path: str | Path | None = None) -> ListenerConfig:
    p = Path(path) if path else configs_dir() / "dicom_listener.yaml"
    return ListenerConfig(**(_load_yaml(p) if p.exists() else {}))


def api_analyzer(cfg: ListenerConfig) -> Callable[[bytes], dict]:
    """POST the Part-10 bytes to ``/api/analyze``; the analysis id and summary come back."""
    import httpx

    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise ValueError(f"dicom_listener: ${cfg.api_key_env} is not set - the API needs a key (D98)")

    def analyze(data: bytes) -> dict:
        r = httpx.post(f"{cfg.api_url.rstrip('/')}/api/analyze", timeout=cfg.api_timeout_s,
                       headers={"X-API-Key": key},
                       files=[("files", ("ecg.dcm", data, "application/dicom"))])
        if r.status_code != 200:
            raise ValueError(f"API {r.status_code}: {r.text[:200]}")
        d = r.json()
        return {"analysis_id": d.get("analysis_id"),
                "heart_rate_bpm": (d.get("measurements") or {}).get("heart_rate_bpm")}

    return analyze


class EcgListener:
    """The receiver. ``start()`` runs it in the background; ``stop()`` shuts it down."""

    def __init__(self, cfg: ListenerConfig, analyze: Callable[[bytes], dict] | None = None):
        if not cfg.allowed_calling_aes:
            raise ValueError("dicom_listener: allowed_calling_aes is empty - no device may connect, "
                             "refusing to start (configs/dicom_listener.yaml)")
        self.cfg = cfg
        self.analyze = analyze or api_analyzer(cfg)
        root = Path(cfg.log_path)
        self.log_path = root if root.is_absolute() else _repo_root() / root
        self._lock = threading.Lock()
        self._server = None
        self._received: dict[int, int] = {}     # association -> bytes received so far

    def _record(self, entry: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def handle_store(self, event) -> int:
        calling = str(event.assoc.requestor.ae_title).strip()
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device_ae": calling,
                 "sop_class": str(event.request.AffectedSOPClassUID),
                 "instance": hashlib.sha256(str(event.request.AffectedSOPInstanceUID).encode()).hexdigest()[:16]}
        try:
            data = event.encoded_dataset(include_meta=True)
        except Exception as exc:  # a malformed object is the device's failure, not ours
            entry.update(status="refused", error=f"cannot encode: {type(exc).__name__}")
            self._record(entry)
            return STATUS_CANNOT_UNDERSTAND
        if len(data) > self.cfg.max_bytes:
            entry.update(status="refused", error=f"too large ({len(data)} bytes)")
            self._record(entry)
            return STATUS_OUT_OF_RESOURCES
        try:
            result = self.analyze(data)
        except Exception as exc:  # reported to the device and logged without content
            entry.update(status="refused", error=str(exc)[:200])
            self._record(entry)
            log.warning("ECG from %s refused: %s", calling, type(exc).__name__)
            return STATUS_CANNOT_UNDERSTAND
        entry.update(status="analysed", **{k: v for k, v in result.items() if k in ("analysis_id", "heart_rate_bpm")})
        self._record(entry)
        return STATUS_OK

    def handle_requested(self, event) -> None:
        """A device outside ``allowed_ips`` is cut off before it can send anything."""
        ip = str(event.assoc.requestor.address)
        if self.cfg.allowed_ips and ip not in self.cfg.allowed_ips:
            self._record({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "refused",
                          "error": "address not allowed"})
            log.warning("DICOM association from a non-allowed address refused")
            event.assoc.abort()

    def handle_data(self, event) -> None:
        """Abort an association past ``max_bytes``: pynetdicom holds the whole dataset in
        memory until C-STORE, so the size check there came after any amount was received."""
        key = id(event.assoc)
        with self._lock:
            n = self._received.get(key, 0) + len(event.data)
            self._received[key] = n
        if n > self.cfg.max_bytes + self.cfg.max_pdu_bytes:
            self._record({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "refused",
                          "device_ae": str(event.assoc.requestor.ae_title).strip(),
                          "error": f"over {self.cfg.max_bytes} bytes; association aborted"})
            event.assoc.abort()

    def handle_released(self, event) -> None:
        with self._lock:
            self._received.pop(id(event.assoc), None)

    def _ae(self):
        from pynetdicom import AE, evt, sop_class
        from pynetdicom.sop_class import Verification

        ae = AE(ae_title=self.cfg.ae_title)
        for name in ECG_STORAGE:
            ae.add_supported_context(getattr(sop_class, name))
        ae.add_supported_context(Verification)
        ae.require_calling_aet = list(self.cfg.allowed_calling_aes)
        ae.maximum_pdu_size = self.cfg.max_pdu_bytes
        ae.maximum_associations = self.cfg.max_associations
        return ae, [(evt.EVT_C_STORE, self.handle_store), (evt.EVT_REQUESTED, self.handle_requested),
                    (evt.EVT_DATA_RECV, self.handle_data), (evt.EVT_RELEASED, self.handle_released),
                    (evt.EVT_ABORTED, self.handle_released)]

    def start(self) -> None:
        ae, handlers = self._ae()
        self._server = ae.start_server((self.cfg.host, self.cfg.port), block=False, evt_handlers=handlers)
        log.info("ECG DICOM listener %s on %s:%d", self.cfg.ae_title, self.cfg.host, self.cfg.port)

    def serve_forever(self) -> None:
        ae, handlers = self._ae()
        ae.start_server((self.cfg.host, self.cfg.port), block=True, evt_handlers=handlers)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
