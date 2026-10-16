"""Digitisers under test: a common interface over our pipeline and the baselines.

* ``cardiolens-v0`` — the existing CardioLens digitiser (``digitize_auto_to_signal``,
  D35–D72) run blind, as a user would: it assumes 25 mm/s, reads px/mm and mm/mV
  from the page and picks 3x4 vs 12x1 from the ink.
* ``cardiolens-v0-oracle`` — the same, told the true paper speed and gain: the gap
  between the two isolates calibration errors from segmentation/vectorisation ones.
* external tools (SD-012) — separate processes writing ``result.json`` (ScanResult).
* from M2 on, ``ecgscan`` itself.

Every digitiser returns a ``ScanResult`` (or an error), so the metrics never care
who produced it.
"""

from __future__ import annotations

import ast
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ecgscan.bench.render import resample_to
from ecgscan.contract import Calibration, Layout, QualitySummary, ScanResult, Source, Trace
from ecgscan.io_utils import sha256_file

GAINS = (5, 10, 20)


@dataclass
class DigitizerRun:
    result: ScanResult | None
    error: str | None
    elapsed_s: float
    extra: dict = field(default_factory=dict)   # raw estimates the contract snaps (e.g. gain 9.7)


class Digitizer(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def digitize(self, image: Path, *, truth_hint: dict | None = None) -> DigitizerRun: ...


def _parse_calibration(notes: list[str]) -> dict:
    for n in notes:
        if n.startswith("calibration: "):
            try:
                return ast.literal_eval(n[len("calibration: "):])
            except (ValueError, SyntaxError):
                return {}
    return {}


def _layout_name(notes: list[str]) -> str | None:
    text = " ".join(notes)
    if "3x4 page" in text:
        return "3x4+rhythm"
    for n in notes:
        if " strip" in n and "x1" in n:
            return n.split()[0]
    return None


def signal_to_scan_result(sig, image: Path, *, fs_out: float, speed: float, speed_source, gain_source,
                          layout: str | None) -> tuple[ScanResult, dict]:
    """A CardioLens ``SignalRepresentation`` from the legacy digitiser -> ``ScanResult``.

    Panels come from ``provenance.lead_windows``; a window markedly longer than the
    others (the 10-s lead II under a 3x4 page) is a rhythm strip.
    """
    cal = _parse_calibration(sig.provenance.notes)
    fs = float(sig.sampling_rate_hz)
    windows = sig.provenance.lead_windows or {n: (0.0, sig.duration_s) for n in sig.lead_names}
    lengths = np.array([t1 - t0 for t0, t1 in windows.values()])
    typical = float(np.median(lengths)) if lengths.size else 0.0
    leads, rhythm = {}, []
    q = float(sig.provenance.confidence or 0.0)
    for lead, (t0, t1) in windows.items():
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        mv = resample_to(sig.lead(lead)[i0:i1].astype(np.float64) / 1000.0, fs, fs_out)
        if not len(mv):
            continue
        tr = Trace.from_array(mv, start_s=t0, fs=fs_out, lead=lead, quality=q, method="dp_path")
        if t1 - t0 > 1.5 * typical:
            rhythm.append(tr)
        else:
            leads[lead] = tr
    gain_raw = cal.get("mm_per_mv")
    gain = min(GAINS, key=lambda g: abs(g - gain_raw)) if gain_raw else 10
    ppm = cal.get("px_per_mm")
    res = ScanResult(
        scan_id=str(uuid.uuid4()), status="ok",
        source=Source(modality="photo", original_sha256=sha256_file(image), client="cli",
                      pipeline_version="cardiolens-v0"),
        calibration=Calibration(px_per_mm_x=ppm, px_per_mm_y=ppm, paper_speed_mm_s=int(speed), gain_mm_mv=gain,
                                speed_source=speed_source,
                                gain_source=gain_source or ("calibration_pulse" if cal.get("pulse_found") else None),
                                confidence=0.0),
        layout=Layout(name=layout, confidence=0.0), sampling_rate_hz=fs_out, leads=leads, rhythm=rhythm,
        quality=QualitySummary(overall=round(q, 3), px_per_mm=ppm),
    )
    return res, {"gain_raw": gain_raw, "px_per_mm_raw": ppm, "pulse_found": cal.get("pulse_found")}


class CardiolensV0:
    """The existing digitiser, blind (``oracle=False``) or told speed and gain."""

    def __init__(self, *, oracle: bool = False, fs_out: float = 500.0):
        self.oracle = oracle
        self.fs_out = fs_out
        self.name = "cardiolens-v0-oracle" if oracle else "cardiolens-v0"

    def available(self) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401

            from cardiolens.layer1_ingestion import digitize_auto_to_signal  # noqa: F401
        except ImportError as exc:
            return False, f"import failed: {exc}"
        return True, ""

    def digitize(self, image: Path, *, truth_hint: dict | None = None) -> DigitizerRun:
        from cardiolens.layer1_ingestion import digitize_auto_to_signal

        kw: dict = {"px_per_mm": None, "mm_per_mv": None}
        speed, speed_src, gain_src = 25.0, None, None
        if self.oracle and truth_hint:
            speed, kw["mm_per_mv"] = float(truth_hint["speed_mm_s"]), float(truth_hint["gain_mm_mv"])
            speed_src = gain_src = "user"
        kw["mm_per_s"] = speed
        t = time.perf_counter()
        try:
            sig = digitize_auto_to_signal(str(image), **kw)
        except Exception as exc:  # the legacy digitiser refuses by raising: that is a result too
            return DigitizerRun(None, f"{type(exc).__name__}: {exc}"[:200], time.perf_counter() - t)
        elapsed = time.perf_counter() - t
        res, extra = signal_to_scan_result(sig, image, fs_out=self.fs_out, speed=speed, speed_source=speed_src,
                                           gain_source=gain_src, layout=_layout_name(sig.provenance.notes))
        return DigitizerRun(res, None, elapsed, extra)


class ExternalCommand:
    """A baseline run as a separate process (SD-012): licence-isolated, contract-only coupling.

    The command template (from the environment variable named in bench.yaml) gets
    ``{image}`` and ``{out_dir}``; the tool (or its wrapper script) must write
    ``{out_dir}/result.json`` as a ScanResult.
    """

    def __init__(self, name: str, cfg: dict, timeout_s: float = 120.0):
        self.name = name
        self.cfg = cfg
        self.timeout_s = timeout_s

    def available(self) -> tuple[bool, str]:
        if not self.cfg.get("enabled"):
            return False, f"disabled in bench.yaml (licence: {self.cfg.get('licence', '?')})"
        if not os.environ.get(self.cfg.get("env", "")):
            return False, f"${self.cfg.get('env')} is not set"
        return True, ""

    def digitize(self, image: Path, *, truth_hint: dict | None = None) -> DigitizerRun:
        template = os.environ[self.cfg["env"]]
        with tempfile.TemporaryDirectory(prefix=f"ecgscan_{self.name}_") as out_dir:
            cmd = [part.format(image=str(image), out_dir=out_dir) for part in shlex.split(template)]
            t = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout_s, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return DigitizerRun(None, f"{type(exc).__name__}: {exc}"[:200], time.perf_counter() - t)
            elapsed = time.perf_counter() - t
            out = Path(out_dir) / "result.json"
            if proc.returncode != 0 or not out.exists():
                err = proc.stderr.decode(errors="replace")[-200:]
                return DigitizerRun(None, f"exit {proc.returncode}: {err}", elapsed)
            try:
                return DigitizerRun(ScanResult.model_validate_json(out.read_text(encoding="utf-8")), None, elapsed)
            except ValueError as exc:
                return DigitizerRun(None, f"invalid result.json: {exc}"[:200], elapsed)


def default_digitizers(cfg: dict) -> list:
    out: list = [CardiolensV0(), CardiolensV0(oracle=True)]
    for name, ext in (cfg.get("external") or {}).items():
        out.append(ExternalCommand(name, ext))
    return out
