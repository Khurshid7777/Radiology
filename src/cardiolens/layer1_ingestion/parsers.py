"""Layer 1 — file parsers to the unified representation (TZ §4).

Working parsers for WFDB, EDF/EDF+, CSV, DICOM waveform ECG (D132) and SCP-ECG
rhythm data (D133). GE-MUSE / Philips / HL7-aECG are declared as interfaces with explicit
``NotImplementedError`` + TODO — we never fabricate a result (TZ §0). All
outputs are in microvolts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Patient,
    Provenance,
    SignalRepresentation,
    canonical_lead_name,
)

_AGE_RE = re.compile(r"\bage>?\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SEX_RE = re.compile(r"\bsex>?\s*:\s*([a-z]+)", re.IGNORECASE)


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


# Physical units to microvolts (EDF / WFDB / CSV).
_UNITS_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "mv": 1000.0, "v": 1e6}


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

    units = [u.strip().lower() if u else "mv" for u in (rec.units or [])]
    # Convert per-lead to µV based on declared units; an unknown unit is refused (D135).
    out = np.empty_like(sig)
    for i in range(sig.shape[0]):
        unit = units[i] if i < len(units) else "mv"
        unit = "mv" if unit == "millivolt" else unit
        if unit not in _UNITS_UV:
            raise ValueError(f"WFDB signal {rec.sig_name[i] if rec.sig_name else i}: unknown unit '{unit}'")
        out[i] = sig[i] * _UNITS_UV[unit]

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
    """Parse an EDF/EDF+ file (TZ §4) to microvolts, each signal by its own physical
    dimension (uV / mV / V) (D97; D119 on our side).

    The ECG leads are the signals labelled as standard leads (all signals when none
    is). Every ECG lead must share one sampling rate and a known unit — otherwise the
    file is refused: a 250-Hz lead read as 500 Hz plays twice as fast, a uV lead read
    as mV is 1000x too large (false ST elevation). Other signals (respiration, SpO2)
    are left out with a note.
    """
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        n = f.signals_in_file
        labels = [canonical_lead_name(x) for x in f.getSignalLabels()]
        rates = [float(f.getSampleFrequency(i)) for i in range(n)]
        dims = [str(f.getPhysicalDimension(i)).strip().lower() for i in range(n)]
        ecg = [i for i in range(n) if labels[i] in STANDARD_12_LEADS] or list(range(n))
        if not ecg:
            raise ValueError("EDF file has no signals")
        if len({rates[i] for i in ecg}) > 1:
            raise ValueError("EDF ECG leads have different sampling rates: "
                             + ", ".join(f"{labels[i]} {rates[i]:g} Hz" for i in ecg))
        unknown = [f"{labels[i]} '{dims[i] or 'none'}'" for i in ecg if dims[i] not in _UNITS_UV]
        if unknown:
            raise ValueError("EDF ECG leads in unknown units (expected uV/mV/V): " + ", ".join(unknown))
        data = [f.readSignal(i) * _UNITS_UV[dims[i]] for i in ecg]
    fs = rates[ecg[0]]
    min_len = min(len(d) for d in data)
    sig = np.vstack([d[:min_len] for d in data]).astype(np.float32)
    left = [labels[i] for i in range(n) if i not in ecg]
    return SignalRepresentation(
        signal=sig,
        sampling_rate_hz=fs,
        lead_names=[labels[i] for i in ecg],
        duration_s=float(min_len / fs),
        source="edf",
        provenance=Provenance(digitized=False, confidence=1.0,
                              notes=[f"EDF signals left out (not ECG leads): {', '.join(left)}"] if left else []),
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
    ``unit`` in {'uv','mv','v'} controls scaling to microvolts; any other is refused.
    """
    path = Path(path)
    if not sampling_rate_hz or float(sampling_rate_hz) <= 0:
        raise ValueError("CSV sampling_rate_hz must be positive")
    if unit.lower() not in _UNITS_UV:
        raise ValueError(f"CSV unit '{unit}' is not one of uv / mv / v")
    header_names: list[str] | None = None
    skip = 0
    if has_header:
        lines = path.read_text(encoding="utf-8-sig").splitlines()       # a BOM made lead I '\ufeffI'
        if not lines:
            raise ValueError("CSV file is empty")
        header_names = [h.strip() for h in lines[0].split(delimiter)]
        skip = 1
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skip, dtype=np.float32, encoding="utf-8-sig", ndmin=2)
    sig = arr.T * np.float32(_UNITS_UV[unit.lower()])   # [leads, samples]
    names = lead_names or header_names or [f"ch{i}" for i in range(sig.shape[0])]
    return SignalRepresentation(
        signal=sig.astype(np.float32),
        sampling_rate_hz=float(sampling_rate_hz),
        lead_names=[canonical_lead_name(n) for n in names],
        duration_s=float(sig.shape[1] / sampling_rate_hz),
        source="csv",
        provenance=Provenance(digitized=False, confidence=1.0),
    )


# DICOM waveform storage SOP classes that carry an ECG (PS3.4 B.5).
DICOM_ECG_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.9.1.1": "12-lead ECG",
    "1.2.840.10008.5.1.4.1.1.9.1.2": "general ECG",
    "1.2.840.10008.5.1.4.1.1.9.1.3": "ambulatory ECG",
    "1.2.840.10008.5.1.4.1.1.9.1.4": "general 32-bit ECG",
}
_DICOM_DTYPES = {("SS", 8): np.int8, ("SB", 8): np.int8, ("UB", 8): np.uint8, ("US", 8): np.uint8,
                 ("SS", 16): np.int16, ("US", 16): np.uint16, ("SL", 32): np.int32, ("UL", 32): np.uint32}
_DICOM_UNITS_UV = {"uv": 1.0, "mv": 1000.0, "v": 1e6}
# SCPECG (5.6.3-9-N) and MDC (2:N) lead codes: the same numbers in both schemes.
_DICOM_LEAD_CODES = {"1": "I", "2": "II", "61": "III", "62": "aVR", "63": "aVL", "64": "aVF",
                     "3": "V1", "4": "V2", "5": "V3", "6": "V4", "7": "V5", "8": "V6"}


def is_dicom(content: bytes) -> bool:
    """Part-10 DICOM file: the ``DICM`` magic after the 128-byte preamble."""
    return len(content) >= 132 and content[128:132] == b"DICM"


def _dicom_lead_name(channel) -> str:
    import re

    src = (channel.get("ChannelSourceSequence") or [None])[0]
    if src is None:
        return str(channel.get("ChannelLabel", "")) or "unknown"
    meaning = str(src.get("CodeMeaning", ""))
    m = re.search(r"\bLead\s+(I{1,3}|aV[RLF]|V[1-6])\b", meaning, flags=re.IGNORECASE)
    if m:
        return canonical_lead_name(m.group(1))
    code = str(src.get("CodeValue", ""))
    m = re.fullmatch(r"(?:5\.6\.3-9-|2:)(\d+)", code)
    if m and m.group(1) in _DICOM_LEAD_CODES:
        return _DICOM_LEAD_CODES[m.group(1)]
    return canonical_lead_name(meaning or code or str(channel.get("ChannelLabel", "unknown")))


def _dicom_age(value) -> float | None:
    """PatientAge ``nnnD|W|M|Y`` in years."""
    s = str(value or "").strip()
    if len(s) != 4 or not s[:3].isdigit():
        return None
    return {"D": 1 / 365.25, "W": 7 / 365.25, "M": 1 / 12, "Y": 1.0}.get(s[3].upper(), 0) * int(s[:3]) or None


def load_dicom(path: str | Path) -> SignalRepresentation:
    """DICOM waveform ECG (12-lead, general or ambulatory; PS3.3 C.10.9) to microvolts (D132).

    The multiplex group with the most standard leads is read (on a tie, the longest:
    the rhythm group rather than a median beat). Each sample becomes
    ``(sample + ChannelBaseline) * ChannelSensitivity * ChannelSensitivityCorrectionFactor``
    in the channel's units (uV / mV / V), then microvolts. A channel without a
    sensitivity has no physical scale: the file is refused, never guessed. Only age and
    sex leave the file; name, IDs and dates are not read into the representation.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        ds = pydicom.dcmread(str(path))
    except InvalidDicomError as exc:
        raise ValueError(f"not a readable DICOM file: {exc}") from None
    sop = str(ds.get("SOPClassUID", ""))
    if sop not in DICOM_ECG_SOP_CLASSES:
        raise ValueError(f"DICOM file is not an ECG waveform (SOP class {sop or 'missing'})")
    groups = list(ds.get("WaveformSequence") or [])
    if not groups:
        raise ValueError("DICOM ECG without a waveform sequence")

    def usable(g) -> bool:
        # The header and the channel list must describe the same multiplex; a mismatch
        # used to scramble the interleaving into leads with no error (D135).
        chans = g.get("ChannelDefinitionSequence") or []
        try:
            n_ch, n_s, fs = int(g.NumberOfWaveformChannels), int(g.NumberOfWaveformSamples), float(g.SamplingFrequency)
        except (AttributeError, TypeError, ValueError):
            return False
        return n_ch > 0 and n_ch == len(chans) and n_s > 0 and fs > 0 and "WaveformData" in g

    groups = [g for g in groups if usable(g)]
    if not groups:
        raise ValueError("DICOM ECG has no consistent waveform group (channels, samples, rate, data)")

    def rank(g):
        names = [_dicom_lead_name(c) for c in g.ChannelDefinitionSequence]
        n_std = sum(n in _DICOM_LEAD_CODES.values() for n in names)
        return n_std, int(g.NumberOfWaveformSamples) / float(g.SamplingFrequency)

    g = max(groups, key=rank)
    n_ch, n_s = int(g.NumberOfWaveformChannels), int(g.NumberOfWaveformSamples)
    fs = float(g.SamplingFrequency)
    key = (str(g.get("WaveformSampleInterpretation", "")), int(g.get("WaveformBitsAllocated", 0) or 0))
    if key not in _DICOM_DTYPES:
        raise ValueError(f"unsupported DICOM waveform sample interpretation {key}")
    raw = np.frombuffer(g.WaveformData, dtype=np.dtype(_DICOM_DTYPES[key]).newbyteorder("<"))
    if raw.size < n_ch * n_s:
        raise ValueError("DICOM waveform data shorter than its declared size")
    raw = raw[:n_ch * n_s].reshape(n_s, n_ch).T.astype(np.float64)   # interleaved by sample
    # Samples equal to WaveformPaddingValue are not data (PS3.3 C.10.9.1.4): the -32768 of
    # a padded tail read as -82 mV and a false LVH. They become NaN for check_record.
    pad = g.get("WaveformPaddingValue")
    if pad is not None:
        pad_val = np.frombuffer(bytes(pad) if isinstance(pad, (bytes, bytearray)) else
                                np.asarray(pad).astype(_DICOM_DTYPES[key]).tobytes(),
                                dtype=np.dtype(_DICOM_DTYPES[key]).newbyteorder("<"))[:1]
        padded = raw == float(pad_val[0]) if pad_val.size else np.zeros_like(raw, bool)
    else:
        padded = np.zeros_like(raw, bool)
    names, rows = [], []
    for i, ch in enumerate(g.ChannelDefinitionSequence):
        if ch.get("ChannelSensitivity") in (None, ""):
            raise ValueError(f"DICOM channel {i + 1} has no sensitivity: no physical scale")
        if float(ch.ChannelSensitivity) == 0:
            raise ValueError(f"DICOM channel {i + 1} has a zero sensitivity")
        unit_seq = ch.get("ChannelSensitivityUnitsSequence") or []
        unit = str(unit_seq[0].get("CodeValue", "")).lower() if unit_seq else ""
        if unit not in _DICOM_UNITS_UV:
            raise ValueError(f"DICOM channel {i + 1}: unsupported unit {unit or 'missing'}")
        sens = float(ch.ChannelSensitivity) * float(ch.get("ChannelSensitivityCorrectionFactor", 1.0) or 1.0)
        base = float(ch.get("ChannelBaseline", 0.0) or 0.0)
        rows.append(np.where(padded[i], np.nan, (raw[i] + base) * sens * _DICOM_UNITS_UV[unit]))
        names.append(_dicom_lead_name(ch))
    sex = {"M": "male", "F": "female"}.get(str(ds.get("PatientSex", "")).upper())
    return SignalRepresentation(
        signal=np.vstack(rows).astype(np.float32),
        sampling_rate_hz=fs,
        lead_names=names,
        duration_s=float(n_s / fs),
        source="dicom",
        patient=Patient(age_years=_dicom_age(ds.get("PatientAge")), sex=sex),
        provenance=Provenance(digitized=False, confidence=1.0,
                              notes=[f"DICOM {DICOM_ECG_SOP_CLASSES[sop]}, group of {n_ch} channels"]
                              + ([f"{int(padded.sum())} padding samples marked invalid"] if padded.any() else [])),
    )


# --- Interfaces declared but not implemented in this increment (TZ §0, §17) ---


def load_scp(path: str | Path) -> SignalRepresentation:
    """SCP-ECG rhythm data (D133; ``cardiolens.layer1_ingestion.scp``)."""
    from cardiolens.layer1_ingestion.scp import load_scp as _load_scp

    return _load_scp(path)


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
    if ext in (".scp",):
        return load_scp(p)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return load_from_image(p, **kwargs)
    raise ValueError(f"Unsupported extension: {ext}")
