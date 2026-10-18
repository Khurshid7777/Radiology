"""Scan submission, background processing, doctor confirmation and analysis (TZ §4, §6.3, §12).

Flow: ``submit`` stores the job (and the encrypted original when a vault is
configured) and queues it; a worker thread runs ``ecgscan.scan`` and stores the
``ScanResult``. ``confirm`` records the doctor's decision in the audit log and the
feedback file; a calibration correction re-runs the scan from the original with the
doctor's values as user hints. ``analyze`` hands an ``ok`` scan to CardioLens.

Logs carry the scan id, status and timing only (TZ §9).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ecgscan.config import load_yaml, repo_root
from ecgscan.contract import Confirmation, ScanResult
from ecgscan.io import _mime
from ecgscan_service.store import FINAL, ScanRow, ScanStore, now
from ecgscan_service.vault import Vault

log = logging.getLogger("ecgscan_service")

HINT_KEYS = ("paper_speed_mm_s", "gain_mm_mv", "layout", "client", "device", "header_masked")
DECISIONS = ("accepted", "corrected", "retake")


class ScanError(Exception):
    """A request the service refuses; ``code`` maps to an HTTP status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Settings:
    root: Path
    keep_originals: bool
    retention_days: float
    max_bytes: int
    mime: tuple[str, ...]
    workers: int
    feedback_path: Path
    public_url: str | None
    warm_up: bool = False

    @classmethod
    def load(cls) -> Settings:
        c = load_yaml("service")
        base = repo_root()
        root = Path(os.environ.get("ECGSCAN_SERVICE_DIR") or base / c["storage"]["dir"])
        fb = Path(c["feedback"]["path"])
        return cls(root=root, keep_originals=bool(c["storage"]["keep_originals"]),
                   retention_days=float(c["storage"]["retention_days"]),
                   max_bytes=int(c["upload"]["max_mb"] * 1024 * 1024), mime=tuple(c["upload"]["mime"]),
                   workers=int(c["queue"]["workers"]), feedback_path=fb if fb.is_absolute() else base / fb,
                   public_url=c.get("public_url"), warm_up=bool(c["queue"].get("warm_up", False)))


class ScanService:
    def __init__(self, settings: Settings | None = None, *, scan_fn=None, vault: Vault | None | bool = True):
        self.settings = settings or Settings.load()
        root = self.settings.root
        self.store = ScanStore(root / "scans.sqlite")
        self.artifacts_root = root / "artifacts"
        if vault is True:
            vault = Vault.from_env(root / "originals") if self.settings.keep_originals else None
        self.vault: Vault | None = vault or None
        self._scan = scan_fn
        self._pool = ThreadPoolExecutor(max_workers=self.settings.workers, thread_name_prefix="ecgscan")
        self._inflight: dict[str, bytes] = {}       # originals of queued jobs when no vault keeps them
        self._lock = threading.Lock()
        if self.settings.warm_up and scan_fn is None:
            from ecgscan.pipeline import warm_up

            self._pool.submit(warm_up)      # models loaded before the first photo arrives (SD-056)
        self._resume()
        self.purge_expired()

    # -- submission and processing ---------------------------------------------------------

    def submit(self, data: bytes, *, owner: str, hints: dict | None = None) -> str:
        if len(data) > self.settings.max_bytes:
            raise ScanError(413, f"file too large (max {self.settings.max_bytes // (1024 * 1024)} MB)")
        if _mime(data) not in self.settings.mime:
            raise ScanError(415, "unsupported file type; expected " + ", ".join(self.settings.mime))
        import hashlib

        scan_id = str(uuid.uuid4())
        hints = {k: v for k, v in (hints or {}).items() if k in HINT_KEYS and v is not None}
        self.store.create(scan_id, owner=owner, sha256=hashlib.sha256(data).hexdigest(), hints=hints)
        self.store.audit(scan_id, owner, "submitted", {"bytes": len(data), "hints": hints})
        if self.vault is not None:
            self.vault.put(scan_id, data)
        else:
            with self._lock:
                self._inflight[scan_id] = data
        self._pool.submit(self._process, scan_id)
        return scan_id

    def _original(self, scan_id: str) -> bytes | None:
        with self._lock:
            data = self._inflight.get(scan_id)
        if data is None and self.vault is not None:
            data = self.vault.get(scan_id)
        return data

    def _process(self, scan_id: str, confirmation: Confirmation | None = None) -> None:
        row = self.store.get(scan_id)
        if row is None:
            return
        data = self._original(scan_id)
        if data is None:
            self.store.update(scan_id, status="failed", error="original not retained")
            return
        self.store.update(scan_id, status="processing")
        t0 = time.perf_counter()
        try:
            res = self._run_scan(data, row.hints, scan_id)
            res.confirmation = confirmation       # a correction is stored with its result, in one write
            self.store.update(scan_id, status=res.status, result=json.loads(res.model_dump_json()))
            log.info("scan %s -> %s in %.1f s", scan_id, res.status, time.perf_counter() - t0)
        except Exception as exc:  # the pipeline maps known failures to codes; this is the unknown rest
            self.store.update(scan_id, status="failed", error=type(exc).__name__)
            log.warning("scan %s failed: %s", scan_id, type(exc).__name__)
        finally:
            with self._lock:
                self._inflight.pop(scan_id, None)

    def _run_scan(self, data: bytes, hints: dict, scan_id: str) -> ScanResult:
        from ecgscan.pipeline import ScanHints, scan

        run = self._scan or scan
        art = self.artifacts_root / scan_id
        if art.exists():
            shutil.rmtree(art)
        res = run(data, hints=ScanHints(**{k: hints.get(k) for k in HINT_KEYS if k in hints}), artifacts_dir=art)
        res.scan_id = scan_id
        # Artifact paths are served by name through the API, never as server paths.
        res.artifacts = {k: Path(v).name for k, v in (res.artifacts or {}).items()}
        return res

    def _resume(self) -> None:
        """After a restart: re-queue what was pending when its original survives (SD-005)."""
        for scan_id in self.store.pending():
            if self.vault is not None and self.vault.has(scan_id):
                self.store.update(scan_id, status="queued")
                self._pool.submit(self._process, scan_id)
            else:
                self.store.update(scan_id, status="failed", error="lost on restart (original not retained)")

    def wait(self, scan_id: str, timeout_s: float = 60.0) -> ScanRow:
        """Block until the scan leaves the queue (tests, the bot)."""
        end = time.monotonic() + timeout_s
        while True:
            row = self.get(scan_id)
            if row.status in FINAL or time.monotonic() > end:
                return row
            time.sleep(0.1)

    def get(self, scan_id: str) -> ScanRow:
        row = self.store.get(scan_id)
        if row is None:
            raise ScanError(404, "scan not found")
        return row

    # -- doctor confirmation (TZ §6.3, §12) -------------------------------------------------

    def confirm(self, scan_id: str, *, actor: str, decision: str, paper_speed_mm_s: float | None = None,
                gain_mm_mv: float | None = None, layout: str | None = None, comment: str | None = None) -> ScanRow:
        row = self.get(scan_id)
        if decision not in DECISIONS:
            raise ScanError(422, f"decision must be one of {DECISIONS}")
        if row.result is None:
            raise ScanError(409, f"scan is {row.status}; nothing to confirm")
        res = ScanResult.model_validate(row.result)
        before = {"paper_speed_mm_s": res.calibration.paper_speed_mm_s, "gain_mm_mv": res.calibration.gain_mm_mv,
                  "layout": res.layout.name, "status": res.status, "warnings": res.warnings}
        changes = {k: v for k, v in (("paper_speed_mm_s", paper_speed_mm_s), ("gain_mm_mv", gain_mm_mv),
                                     ("layout", layout)) if v is not None}
        conf = Confirmation(by=actor, at=now(), decision=decision, changes={"before": before, "after": changes})
        if decision == "accepted":
            if res.status == "rejected":
                raise ScanError(409, "a rejected scan cannot be accepted; retake it")
            if res.calibration.paper_speed_mm_s is None or res.calibration.gain_mm_mv is None:
                raise ScanError(422, "speed and gain are not established: confirm them with decision 'corrected'")
            res.status = "ok"
            res.confirmation = conf
            self.store.update(scan_id, status="ok", result=json.loads(res.model_dump_json()))
        elif decision == "corrected":
            if not changes:
                raise ScanError(422, "a correction needs paper_speed_mm_s, gain_mm_mv or layout")
            data = self._original(scan_id)
            if data is None:
                raise ScanError(409, "the original is not retained (no storage key): submit the photo again")
            hints = {**row.hints, **changes}
            self.store.update(scan_id, status="queued", hints=hints)
            self._pool.submit(self._process, scan_id, conf)
        else:  # retake
            res.confirmation = conf
            self.store.update(scan_id, status=res.status, result=json.loads(res.model_dump_json()))
        self.store.audit(scan_id, actor, f"confirm:{decision}", {"changes": changes, "comment": comment})
        self._feedback(row, conf, comment)
        return self.get(scan_id)

    def _feedback(self, row: ScanRow, conf: Confirmation, comment: str | None) -> None:
        """Doctor decisions as training feedback (TZ §12): calibration before/after, no images."""
        p = self.settings.feedback_path
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"scan_id": row.id, "sha256": row.sha256, "by": conf.by, "at": conf.at, "decision": conf.decision,
               "changes": conf.changes, "comment": comment}
        with self._lock, p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- CardioLens (TZ §12) -----------------------------------------------------------------

    def analyze(self, scan_id: str, *, actor: str, mode: str | None = None) -> dict:
        row = self.get(scan_id)
        if row.status != "ok" or row.result is None:
            raise ScanError(409, f"scan is {row.status}: only an ok or doctor-accepted scan is analysed")
        from cardiolens.api import service as cl_service
        from cardiolens.outputs.conclusion import format_conclusion
        from ecgscan.contract.cardiolens_adapter import to_signal_representation

        sig = to_signal_representation(ScanResult.model_validate(row.result))
        analysis_id, result = cl_service.run_and_store(sig, mode=mode)
        self.store.set_analysis(scan_id, analysis_id)
        self.store.audit(scan_id, actor, "analyzed", {"analysis_id": analysis_id, "mode": result.report.mode})
        return {"analysis_id": analysis_id, "mode": result.report.mode,
                "conclusion_ru": format_conclusion(result, "ru"), "conclusion_uz": format_conclusion(result, "uz"),
                "report_pdf": f"/api/analysis/{analysis_id}/report.pdf"}

    # -- privacy -----------------------------------------------------------------------------

    def artifact_path(self, scan_id: str, name: str) -> Path:
        row = self.get(scan_id)
        files = (row.result or {}).get("artifacts") or {}
        if name not in files:
            raise ScanError(404, "artifact not found")
        p = self.artifacts_root / scan_id / files[name]
        if not p.exists():
            raise ScanError(410, "artifact expired")
        return p

    def purge_expired(self, at: datetime | None = None) -> int:
        """Delete originals and artifact images past the retention period (TZ §9)."""
        cutoff = ((at or datetime.now(UTC)) - timedelta(days=self.settings.retention_days)).isoformat(
            timespec="seconds")
        ids = self.store.older_than(cutoff)
        for scan_id in ids:
            if self.vault is not None:
                self.vault.delete(scan_id)
            shutil.rmtree(self.artifacts_root / scan_id, ignore_errors=True)
        return len(ids)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
