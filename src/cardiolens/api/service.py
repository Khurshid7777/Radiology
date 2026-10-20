"""Request handling helpers for the API — build a signal, run the pipeline.

Kept separate from the FastAPI wiring so it is unit-testable without HTTP.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from cardiolens.layer1_ingestion import (
    generate_synthetic_ecg,
    is_dicom,
    is_scp,
    load_csv,
    load_dicom,
    load_edf,
    load_scp,
    load_wfdb,
)
from cardiolens.pipeline import PipelineResult, run_pipeline

# Guardrails for the public-facing service.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024   # 64 MB total across uploaded files
STORE_MAX_ENTRIES = 200               # bounded in-memory analysis cache

# Synthetic demo cases exposed by the web UI (mirrors cardiolens.demo).
DEMO_CASES: dict[str, dict] = {
    "normal": dict(heart_rate_bpm=72),
    "brady": dict(heart_rate_bpm=44),
    "tachy": dict(heart_rate_bpm=125),
    "hyperk": dict(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6),
    "stemi": dict(heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]),
    "child": dict(heart_rate_bpm=95, age_years=6),
}


@dataclass
class UploadFile:
    filename: str
    content: bytes


def signal_from_case(case: str):
    if case not in DEMO_CASES:
        raise ValueError(f"unknown case '{case}'")
    return generate_synthetic_ecg(**DEMO_CASES[case])


def signal_from_uploads(
    files: list[UploadFile],
    *,
    sampling_rate_hz: float | None = None,
    unit: str = "uv",
):
    """Persist uploaded files to a temp dir and dispatch to a parser.

    Supports WFDB (.hea + .dat), EDF (.edf/.bdf), CSV (.csv/.txt) and DICOM waveform
    ECG (.dcm, or any file with the DICOM magic: devices export without an extension) and
    SCP-ECG (.scp, or a record whose CRC and section-0 mark hold).
    """
    if not files:
        raise ValueError("no files uploaded")
    total = sum(len(f.content) for f in files)
    if total > MAX_UPLOAD_BYTES:
        raise ValueError(f"upload too large ({total} bytes > {MAX_UPLOAD_BYTES})")
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_api_"))
    try:
        saved: list[Path] = []
        for f in files:
            # Path(...).name strips any directory components (traversal-safe).
            p = tmp / Path(f.filename).name
            p.write_bytes(f.content)
            saved.append(p)

        # A malformed file is the sender's error (400 with the reason), whatever the
        # parser raised underneath: truncated SCP sections, junk EDF, a 0-d CSV (D97).
        try:
            dcm = next((p for p, f in zip(saved, files, strict=True)
                        if p.suffix.lower() == ".dcm" or is_dicom(f.content)), None)
            if dcm is not None:
                return load_dicom(dcm)
            scp = next((p for p, f in zip(saved, files, strict=True)
                        if p.suffix.lower() == ".scp" or is_scp(f.content)), None)
            if scp is not None:
                return load_scp(scp)
            hea = next((p for p in saved if p.suffix.lower() == ".hea"), None)
            if hea is not None:
                return load_wfdb(hea.with_suffix(""))
            edf = next((p for p in saved if p.suffix.lower() in (".edf", ".bdf")), None)
            if edf is not None:
                return load_edf(edf)
            csv = next((p for p in saved if p.suffix.lower() in (".csv", ".txt")), None)
            if csv is not None:
                if sampling_rate_hz is None:
                    raise ValueError("sampling_rate_hz is required for CSV input")
                return load_csv(csv, sampling_rate_hz=sampling_rate_hz, unit=unit)
            raise ValueError("unsupported upload; expected WFDB(.hea+.dat), EDF, CSV, DICOM ECG or SCP-ECG")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"could not read the uploaded ECG file ({type(exc).__name__})") from exc
    finally:
        # Parsers read the raw signal into memory, so the temp files are no
        # longer needed once parsing returns (or raises) — never leak them.
        shutil.rmtree(tmp, ignore_errors=True)


def apply_patient_overrides(
    sig,
    *,
    age_years: float | None = None,
    age_days: float | None = None,
    sex: str | None = None,
    potassium: float | None = None,
    calcium: float | None = None,
):
    if age_years is not None:
        sig.patient.age_years = age_years
    if age_days is not None:
        sig.patient.age_days = age_days
    if sex is not None:
        sig.patient.sex = sex
    if potassium is not None:
        sig.context.potassium = potassium
    if calcium is not None:
        sig.context.calcium = calcium
    return sig


# --- Bounded in-memory analysis store (MVP; not for multi-worker production) ---
# LRU-evicting so a long-running server can't grow without bound. Swap for a DB
# / object store in production (see DECISIONS.md).
_STORE: OrderedDict[str, PipelineResult] = OrderedDict()
_OWNERS: dict[str, str | None] = {}      # analysis id -> API key name that created it (D98)


def run_and_store(sig, *, mode: str | None, prior_id: str | None = None,
                  owner: str | None = None) -> tuple[str, PipelineResult]:
    prior = get_analysis(prior_id, owner=owner) if prior_id else None
    if prior_id:
        sig.context.prior_ecg_id = prior_id
    result = run_pipeline(sig, mode=mode, prior=prior)
    analysis_id = uuid.uuid4().hex[:12]
    _STORE[analysis_id] = result
    _OWNERS[analysis_id] = owner
    _STORE.move_to_end(analysis_id)
    while len(_STORE) > STORE_MAX_ENTRIES:
        old_id, _ = _STORE.popitem(last=False)  # evict least-recently-used
        _OWNERS.pop(old_id, None)
    return analysis_id, result


def get_analysis(analysis_id: str, *, owner: str | None = None) -> PipelineResult | None:
    """``owner``: the caller's key name; another key's analysis reads as absent (D98)."""
    if owner is not None and _OWNERS.get(analysis_id) not in (None, owner):
        return None
    result = _STORE.get(analysis_id)
    if result is not None:
        _STORE.move_to_end(analysis_id)  # mark as recently used
    return result
