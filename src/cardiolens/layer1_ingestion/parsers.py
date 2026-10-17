"""Layer 1 — file parsers to the unified representation (TZ §4).

Working parsers for WFDB, EDF/EDF+ and CSV (the clean-digital MVP path,
TZ §17). DICOM-ECG / SCP-ECG / GE-MUSE / Philips / HL7-aECG are declared as
interfaces with explicit ``NotImplementedError`` + TODO — we never fabricate a
result (TZ §0). All outputs are in microvolts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import (
    Patient,
    Provenance,
    SignalRepresentation,
    canonical_lead_name,
)

_AGE_RE = re.compile(r"\bage\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SEX_RE = re.compile(r"\bsex\s*:\s*([a-z]+)", re.IGNORECASE)


def patient_from_wfdb_comments(comments: list[str] | None) -> Patient:
    """Age and sex from WFDB header comments (D101).

    PhysioNet Challenge 2020/2021 headers carry ``# Age: 74`` / ``# Sex: Female`` lines,
    PTBDB ``# age: 81`` / ``# sex: female``. Unknown or ``NaN`` stays None. Without this the
    rules fell back to male thresholds (QTc 450, STEMI V2-V3, Cornell) for every woman.
    """
    text = " ".join(comments or [])
    age = sex = None
    m = _AGE_RE.search(text)
    if m:
        age = float(m.group(1))
        if not 0 <= age <= 120:
            age = None
    m = _SEX_RE.search(text)
    if m:
        v = m.group(1).lower()
        sex = "female" if v in ("female", "f", "woman") else "male" if v in ("male", "m", "man") else None
    return Patient(age_years=age, sex=sex)


def _mv_to_uv(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * 1000.0).astype(np.float32)


def load_wfdb(record_path: str | Path, *, sampto: int | None = None,
              channels: list[int] | None = None) -> SignalRepresentation:
    """Parse a WFDB record (``.hea``/``.dat``) — MIT-BIH, PTB-XL, PTBDB (TZ §4).

    ``record_path`` is the record name WITHOUT extension (wfdb convention).
    WFDB physical units are typically millivolts -> converted to µV. ``sampto`` /
    ``channels`` cut long multi-channel records (PTBDB: 2 min, 15 leads) on read.
    """
    import wfdb

    record_path = str(record_path)
    rec = wfdb.rdrecord(record_path, sampto=sampto, channels=channels)
    sig = np.asarray(rec.p_signal, dtype=np.float32).T  # -> [leads, samples]

    units = [u.lower() if u else "mv" for u in (rec.units or [])]
    # Convert per-lead to µV based on declared units.
    out = np.empty_like(sig)
    for i in range(sig.shape[0]):
        unit = units[i] if i < len(units) else "mv"
        out[i] = sig[i] * 1000.0 if unit in ("mv", "millivolt") else sig[i]

    lead_names = [canonical_lead_name(n) for n in rec.sig_name] if rec.sig_name else [f"ch{i}" for i in range(sig.shape[0])]
    return SignalRepresentation(
        signal=out,
        sampling_rate_hz=float(rec.fs),
        lead_names=lead_names,
        duration_s=float(sig.shape[1] / rec.fs),
        source="wfdb",
        patient=patient_from_wfdb_comments(getattr(rec, "comments", None)),
        provenance=Provenance(digitized=False, confidence=1.0),
    )


def load_edf(path: str | Path) -> SignalRepresentation:
    """Parse an EDF/EDF+ file (TZ §4). Assumes signals in mV -> µV."""
    import pyedflib

    path = str(path)
    with pyedflib.EdfReader(path) as f:
        n = f.signals_in_file
        labels = f.getSignalLabels()
        fs = float(f.getSampleFrequency(0))
        data = [f.readSignal(i) for i in range(n)]
    min_len = min(len(d) for d in data)
    sig = np.vstack([d[:min_len] for d in data]).astype(np.float32)
    sig = _mv_to_uv(sig)
    return SignalRepresentation(
        signal=sig,
        sampling_rate_hz=fs,
        lead_names=[canonical_lead_name(n) for n in labels],
        duration_s=float(min_len / fs),
        source="edf",
        provenance=Provenance(digitized=False, confidence=1.0),
    )


def load_csv(
    path: str | Path,
    *,
    sampling_rate_hz: float,
    lead_names: list[str] | None = None,
    unit: str = "uv",
    delimiter: str = ",",
    has_header: bool = True,
) -> SignalRepresentation:
    """Parse a CSV of samples (columns = leads) (TZ §4).

    Sampling rate is REQUIRED (not stored in CSV) — never guessed (TZ §0).
    ``unit`` in {'uv','mv'} controls scaling to microvolts.
    """
    path = Path(path)
    header_names: list[str] | None = None
    skip = 0
    if has_header:
        header_names = path.read_text(encoding="utf-8").splitlines()[0].split(delimiter)
        skip = 1
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skip, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    sig = arr.T  # [leads, samples]
    if unit.lower() == "mv":
        sig = _mv_to_uv(sig)
    names = lead_names or header_names or [f"ch{i}" for i in range(sig.shape[0])]
    return SignalRepresentation(
        signal=sig.astype(np.float32),
        sampling_rate_hz=float(sampling_rate_hz),
        lead_names=[canonical_lead_name(n) for n in names],
        duration_s=float(sig.shape[1] / sampling_rate_hz),
        source="csv",
        provenance=Provenance(digitized=False, confidence=1.0),
    )


# --- Interfaces declared but not implemented in this increment (TZ §0, §17) ---
def load_dicom(path: str | Path) -> SignalRepresentation:  # pragma: no cover
    raise NotImplementedError("TODO(v2): DICOM-ECG Waveform parsing (TZ §4).")


def load_scp(path: str | Path) -> SignalRepresentation:  # pragma: no cover
    raise NotImplementedError("TODO(v2): SCP-ECG parsing (TZ §4).")


def load_from_image(path: str | Path, **kwargs) -> SignalRepresentation:
    """Digitise a single-lead trace from a clean gridded image (TZ §4).

    Clean scans/PDF renders only in this increment; dirty-photo robustness and
    multi-lead segmentation are v2 (TZ §17). Delegates to ``digitize_image``.
    """
    from cardiolens.layer1_ingestion.digitize import digitize_image

    return digitize_image(path, **kwargs)


def load(path: str | Path, **kwargs) -> SignalRepresentation:
    """Dispatch to a parser by file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".hea", ".dat", ""):
        return load_wfdb(p.with_suffix(""))
    if ext in (".edf", ".bdf"):
        return load_edf(p)
    if ext in (".csv", ".txt"):
        return load_csv(p, **kwargs)
    if ext in (".dcm",):
        return load_dicom(p)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return load_from_image(p, **kwargs)
    raise ValueError(f"Unsupported extension: {ext}")
