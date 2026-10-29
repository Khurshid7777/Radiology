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
from ecgscan.io import _mime, image_size
from ecgscan_service.store import FINAL, ScanRow, ScanStore, now
from ecgscan_service.vault import Vault

log = logging.getLogger("ecgscan_service")

HINT_KEYS = ("paper_speed_mm_s", "gain_mm_mv", "layout", "client", "device", "header_masked", "capture",
             "device_profile", "pages")
DECISIONS = ("accepted", "corrected", "retake")


class ScanError(Exception):
    """A request the service refuses; ``code`` maps to an HTTP status."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


_PARTS_MAGIC = b"ECGSCAN-PARTS1\n"


def pack_parts(parts: list[bytes]) -> bytes:
    """Several photos as one blob (one sealed original, one retention): a magic line,
    the count and each length as 8-byte big-endian, then the photos."""
    head = _PARTS_MAGIC + len(parts).to_bytes(8, "big") + b"".join(len(x).to_bytes(8, "big") for x in parts)
    return head + b"".join(parts)


def unpack_parts(blob: bytes) -> bytes | list[bytes]:
    if not blob.startswith(_PARTS_MAGIC):
        return blob
    pos = len(_PARTS_MAGIC)
    n = int.from_bytes(blob[pos:pos + 8], "big")
    lens = [int.from_bytes(blob[pos + 8 + 8 * i:pos + 16 + 8 * i], "big") for i in range(n)]
    pos += 8 + 8 * n
    out = []
    for ln in lens:
        out.append(blob[pos:pos + ln])
        pos += ln
    if pos != len(blob):
        raise ValueError("damaged parts container")
    return out


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
    max_pending: int = 20               # queued + processing; more -> 429 (D136)
    purge_interval_s: float = 3600.0    # retention is enforced on a timer, not only at start (D136)
    max_comment_chars: int = 500
    admins: tuple[str, ...] = ()        # key names that may open any client's scans
    max_megapixels: float = 60.0
    max_parts: int = 4                  # photos of one page in parts on a capture board (SD-066)

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
                   public_url=c.get("public_url"), warm_up=bool(c["queue"].get("warm_up", False)),
                   max_pending=int(c["queue"].get("max_pending", 20)),
                   purge_interval_s=float(c["storage"].get("purge_interval_h", 1.0)) * 3600.0,
                   max_comment_chars=int(c.get("confirm", {}).get("max_comment_chars", 500)),
                   admins=tuple(c.get("access", {}).get("admins") or ()),
                   max_megapixels=float(load_yaml("pipeline")["quality"].get("max_megapixels", 60)),
                   max_parts=int(c["upload"].get("max_parts", 4)))


class ScanService:
    def __init__(self, settings: Settings | None = None, *, scan_fn=None, vault: Vault | None | bool = True):
        self.settings = settings or Settings.load()
        root = self.settings.root
        self.store = ScanStore(root / "scans.sqlite")
        self.artifacts_root = root / "artifacts"
        if vault is True:
            vault = Vault.from_env(root / "originals") if self.settings.keep_originals else None
            if vault is None and self.settings.keep_originals:
                log.warning("keep_originals is on but ECGSCAN_STORAGE_KEY is not set: originals are not kept, "
                            "corrections and restarts cannot re-run a scan, artifact images stay unencrypted")
        self.vault: Vault | None = vault or None
        self._scan = scan_fn
        self._pool = ThreadPoolExecutor(max_workers=self.settings.workers, thread_name_prefix="ecgscan")
        self._inflight: dict[str, bytes] = {}       # originals of queued jobs when no vault keeps them
        self._lock = threading.Lock()
        self._pending = 0
        self._stop = threading.Event()
        if self.settings.warm_up and scan_fn is None:
            from ecgscan.pipeline import warm_up

            self._pool.submit(warm_up)      # models loaded before the first photo arrives (SD-056)
        self._resume()
        self.purge_expired()
        if self.settings.purge_interval_s > 0:
            threading.Thread(target=self._purge_loop, name="ecgscan-purge", daemon=True).start()

    def _purge_loop(self) -> None:
        while not self._stop.wait(self.settings.purge_interval_s):
            try:
                n = self.purge_expired()
                if n:
                    log.info("retention: %d scan(s) purged", n)
            except Exception as exc:  # the timer must survive a locked database
                log.warning("retention purge failed: %s", type(exc).__name__)

    # -- submission and processing ---------------------------------------------------------

    def submit(self, data: bytes | list[bytes], *, owner: str, hints: dict | None = None) -> str:
        """One photo, or a list: one page photographed in parts on a capture board (SD-066),
        kept as one sealed original (``pack_parts``)."""
        parts = data if isinstance(data, list) else [data]
        if not parts or len(parts) > self.settings.max_parts:
            raise ScanError(400, f"1 to {self.settings.max_parts} photos per scan")
        for part in parts:
            self._check_upload(part)
        with self._lock:
            if self._pending >= self.settings.max_pending:
                raise ScanError(429, "too many scans in the queue; try again shortly")
            self._pending += 1
        import hashlib

        scan_id = str(uuid.uuid4())
        hints = {k: v for k, v in (hints or {}).items() if k in HINT_KEYS and v is not None}
        if len(parts) == 1:
            data, sha = parts[0], hashlib.sha256(parts[0]).hexdigest()
        else:
            # As ScanResult.source: the hash of the parts' hex digests, in the order sent.
            data = pack_parts(parts)
            sha = hashlib.sha256("".join(hashlib.sha256(x).hexdigest() for x in parts).encode()).hexdigest()
        self.store.create(scan_id, owner=owner, sha256=sha, hints=hints)
        self.store.audit(scan_id, owner, "submitted", {"bytes": len(data), "parts": len(parts), "hints": hints})
        if self.vault is not None:
            self.vault.put(scan_id, data)
        else:
            with self._lock:
                self._inflight[scan_id] = data
        self._pool.submit(self._process, scan_id)
        return scan_id

    def _check_upload(self, data: bytes) -> None:
        if len(data) > self.settings.max_bytes:
            raise ScanError(413, f"file too large (max {self.settings.max_bytes // (1024 * 1024)} MB)")
        if _mime(data) not in self.settings.mime:
            raise ScanError(415, "unsupported file type; expected " + ", ".join(self.settings.mime))
        size = image_size(data)
        if size is not None and size[0] * size[1] > self.settings.max_megapixels * 1e6:
            raise ScanError(413, f"image too large ({size[0]} x {size[1]} px; max "
                                 f"{self.settings.max_megapixels:g} MP)")

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
                self._pending = max(0, self._pending - 1)

    def _run_scan(self, data: bytes, hints: dict, scan_id: str) -> ScanResult:
        from ecgscan.pipeline import ScanHints, scan

        run = self._scan or scan
        art = self.artifacts_root / scan_id
        if art.exists():
            shutil.rmtree(art)
        res = run(unpack_parts(data), hints=ScanHints(**{k: hints.get(k) for k in HINT_KEYS if k in hints}), artifacts_dir=art)
        res.scan_id = scan_id
        # Artifact paths are served by name through the API, never as server paths. With a
        # storage key they are kept encrypted like the originals (D136).
        files = {k: Path(v) for k, v in (res.artifacts or {}).items()}
        if self.vault is not None:
            files = {k: self.vault.seal_file(v) for k, v in files.items()}
        res.artifacts = {k: v.name for k, v in files.items()}
        return res

    def _resume(self) -> None:
        """After a restart: re-queue what was pending when its original survives (SD-005)."""
        for scan_id in self.store.pending():
            if self.vault is not None and self.vault.has(scan_id):
                self.store.update(scan_id, status="queued")
                with self._lock:
                    self._pending += 1
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

    def get(self, scan_id: str, actor: str | None = None) -> ScanRow:
        """``actor``: the API key's name; a scan of another client reads as not found (D136)."""
        row = self.store.get(scan_id)
        if row is None or (actor is not None and actor != row.owner and actor not in self.settings.admins):
            raise ScanError(404, "scan not found")
        return row

    # -- doctor confirmation (TZ §6.3, §12) -------------------------------------------------

    def confirm(self, scan_id: str, *, actor: str, decision: str, paper_speed_mm_s: float | None = None,
                gain_mm_mv: float | None = None, layout: str | None = None, comment: str | None = None) -> ScanRow:
        row = self.get(scan_id, actor)
        if comment is not None and len(comment) > self.settings.max_comment_chars:
            raise ScanError(422, f"comment longer than {self.settings.max_comment_chars} characters")
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
            with self._lock:
                self._pending += 1
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
        row = self.get(scan_id, actor)
        if row.status != "ok" or row.result is None:
            raise ScanError(409, f"scan is {row.status}: only an ok or doctor-accepted scan is analysed")
        from cardiolens.api import service as cl_service
        from cardiolens.outputs.conclusion import format_conclusion
        from cardiolens.telegram.analyze import attach_device_header
        from ecgscan.contract.cardiolens_adapter import to_signal_representation

        read = ScanResult.model_validate(row.result)
        sig = to_signal_representation(read)
        if "trace_overlay_preview" in ((row.result or {}).get("artifacts") or {}):
            try:                     # the page with the read traces, header blanked: the PDF shows it (TZ 18.10)
                sig.provenance.overlay_jpeg = self.artifact(scan_id, "trace_overlay_preview", actor)[0]
            except ScanError:        # expired: the report goes without the picture
                pass
        attach_device_header(sig, read)       # the recorder's printed numbers: device_mismatch (D183)
        analysis_id, result = cl_service.run_and_store(sig, mode=mode, owner=row.owner)
        self.store.set_analysis(scan_id, analysis_id)
        self.store.audit(scan_id, actor, "analyzed", {"analysis_id": analysis_id, "mode": result.report.mode})
        return {"analysis_id": analysis_id, "mode": result.report.mode,
                "conclusion_ru": format_conclusion(result, "ru"), "conclusion_uz": format_conclusion(result, "uz"),
                "report_pdf": f"/api/analysis/{analysis_id}/report.pdf"}

    # -- export (TZ 18.10, S4) -----------------------------------------------------------------

    def export(self, scan_id: str, fmt: str, actor: str | None = None) -> tuple[bytes, str, str]:
        """(bytes, media type, file name) of the read signal: the contract JSON, CSV or WFDB."""
        from ecgscan.export import EXPORTS, export

        row = self.get(scan_id, actor)
        if fmt not in EXPORTS:
            raise ScanError(422, f"format must be one of {sorted(EXPORTS)}")
        if row.result is None or not (row.result.get("leads") or row.result.get("rhythm")):
            raise ScanError(409, f"scan is {row.status}; no signal to export")
        data, media, ext = export(ScanResult.model_validate(row.result), fmt)
        self.store.audit(scan_id, actor, "exported", {"format": fmt})
        return data, media, f"ecg-{scan_id[:8]}.{ext}"

    # -- privacy -----------------------------------------------------------------------------

    def artifact(self, scan_id: str, name: str, actor: str | None = None) -> tuple[bytes, str]:
        """(bytes, media type) of an artifact image, decrypted when the vault sealed it."""
        row = self.get(scan_id, actor)
        files = (row.result or {}).get("artifacts") or {}
        if name not in files:
            raise ScanError(404, "artifact not found")
        p = self.artifacts_root / scan_id / files[name]
        if not p.exists():
            raise ScanError(410, "artifact expired")
        plain = p.name.removesuffix(".enc")
        media = "image/jpeg" if plain.endswith(".jpg") else "image/png"
        if p.suffix == ".enc":
            if self.vault is None:
                raise ScanError(410, "artifact is encrypted and the storage key is not configured")
            return self.vault.open_file(p), media
        return p.read_bytes(), media

    def purge_expired(self, at: datetime | None = None) -> int:
        """Delete originals and artifact images past the retention period (TZ §9)."""
        cutoff = ((at or datetime.now(UTC)) - timedelta(days=self.settings.retention_days)).isoformat(
            timespec="seconds")
        commented = self._comment_ids()
        ids = [i for i in self.store.older_than(cutoff)
               if (self.vault is not None and self.vault.has(i)) or (self.artifacts_root / i).exists()
               or i in commented]
        for scan_id in ids:
            if self.vault is not None:
                self.vault.delete(scan_id)
            shutil.rmtree(self.artifacts_root / scan_id, ignore_errors=True)
        if ids:
            self.store.scrub_comments(ids)
            self._scrub_feedback(set(ids))
        return len(ids)

    def _comment_ids(self) -> set[str]:
        p = self.settings.feedback_path
        if not p.exists():
            return set()
        with self._lock:
            lines = p.read_text(encoding="utf-8").splitlines()
        return {json.loads(x)["scan_id"] for x in lines if x.strip() and json.loads(x).get("comment")}

    def _scrub_feedback(self, ids: set[str]) -> None:
        """The feedback file keeps the decision and the calibration change, not the free text."""
        p = self.settings.feedback_path
        if not p.exists():
            return
        with self._lock:
            out = []
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("scan_id") in ids and rec.get("comment"):
                    rec["comment"] = None
                out.append(json.dumps(rec, ensure_ascii=False))
            tmp = p.with_suffix(".tmp")
            tmp.write_text("".join(x + "\n" for x in out), encoding="utf-8")
            tmp.replace(p)

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=True)
