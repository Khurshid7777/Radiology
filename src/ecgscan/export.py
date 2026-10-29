"""Signal export (S4, TZ 18.10): the CardioVision contract (``ScanResult`` JSON) and, for
research, CSV and WFDB.

All channels share one time axis at ``ScanResult.sampling_rate_hz`` from t = 0: the twelve leads
in the standard order, each present only inside its own time window (a 3x4 page shows a lead for
2.5 s), then the rhythm strips (``II_rhythm``, ``V1_rhythm``, ...). A sample the page did not show
is missing (an empty CSV cell, WFDB's invalid value), never zero.

* CSV: ``time_s`` and one column per channel, mV with Python's shortest round-trip repr: read
  back bit for bit.
* WFDB (format 32, 100 000 adu/mV, baseline 0): one step is 0.01 µV, the contract's own precision
  (``Trace.from_array`` keeps 5 decimals of mV), so it also reads back bit for bit; format 16 would
  have rounded to 1 µV. The header comments carry the scan's status, calibration and warnings,
  never personal data.
"""

from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ecgscan.contract.models import STANDARD_LEADS, ScanResult

WFDB_GAIN = 100_000.0     # adu per mV: 0.01 µV steps, the contract's precision
WFDB_FMT = "32"


def channels(result: ScanResult) -> tuple[list[str], np.ndarray]:
    """Channel names and a (n_channels, n_samples) mV array on one axis, NaN where missing."""
    fs = float(result.sampling_rate_hz)
    traces = [(lead, result.leads[lead]) for lead in STANDARD_LEADS if lead in result.leads]
    traces += [(lead, tr) for lead, tr in sorted(result.leads.items()) if lead not in STANDARD_LEADS]
    seen: dict[str, int] = {}
    for tr in result.rhythm:
        base = f"{tr.lead or 'unknown'}_rhythm"
        seen[base] = seen.get(base, 0) + 1
        traces.append((base if seen[base] == 1 else f"{base}{seen[base]}", tr))
    if not traces:
        return [], np.zeros((0, 0))
    starts = [int(round(tr.start_s * fs)) for _, tr in traces]
    n = max(s + len(tr.samples_mv) for s, (_, tr) in zip(starts, traces, strict=True))
    data = np.full((len(traces), n), np.nan)
    for row, (s, (_, tr)) in enumerate(zip(starts, traces, strict=True)):
        a = tr.array()
        data[row, s:s + len(a)] = a
    return [name for name, _ in traces], data


def to_csv(result: ScanResult) -> str:
    names, data = channels(result)
    fs = float(result.sampling_rate_hz)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["time_s", *names])
    for i in range(data.shape[1]):
        w.writerow([repr(round(i / fs, 6)), *("" if np.isnan(v) else repr(float(v)) for v in data[:, i])])
    return buf.getvalue()


def read_csv(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(channel names, time_s, data) from ``to_csv``'s text."""
    rows = list(csv.reader(io.StringIO(text)))
    names = rows[0][1:]
    t = np.array([float(r[0]) for r in rows[1:]])
    data = np.array([[float(v) if v != "" else np.nan for v in r[1:]] for r in rows[1:]]).T
    return names, t, data.reshape(len(names), len(t))


def _comments(result: ScanResult) -> list[str]:
    c = result.calibration
    out = [f"ecgscan {result.source.pipeline_version} scan {result.scan_id} status {result.status}",
           f"speed_mm_s {c.paper_speed_mm_s} ({c.speed_source}) gain_mm_mv {c.gain_mm_mv} ({c.gain_source})",
           f"layout {result.layout.name} source {result.source.modality}", "digitized_from_image"]
    if result.warnings:
        out.append("warnings " + " ".join(result.warnings))
    return out


def to_wfdb_zip(result: ScanResult, record: str = "ecg") -> bytes:
    """A zip holding ``<record>.hea`` and ``<record>.dat``."""
    import wfdb

    names, data = channels(result)
    if not names:
        raise ValueError("no traces to export")
    with tempfile.TemporaryDirectory() as tmp:
        wfdb.wrsamp(record, fs=float(result.sampling_rate_hz), units=["mV"] * len(names), sig_name=names,
                    p_signal=data.T, fmt=[WFDB_FMT] * len(names), adc_gain=[WFDB_GAIN] * len(names),
                    baseline=[0] * len(names), comments=_comments(result), write_dir=tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for ext in ("hea", "dat"):
                z.write(Path(tmp) / f"{record}.{ext}", f"{record}.{ext}")
        return buf.getvalue()


def read_wfdb_zip(data: bytes, record: str = "ecg") -> tuple[list[str], float, np.ndarray, list[str]]:
    """(channel names, fs, (n_channels, n_samples) mV with NaN, header comments)."""
    import wfdb

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        rec = wfdb.rdrecord(str(Path(tmp) / record))
    return list(rec.sig_name), float(rec.fs), rec.p_signal.T, list(rec.comments)


EXPORTS = {
    "json": ("application/json", "json"),
    "csv": ("text/csv; charset=utf-8", "csv"),
    "wfdb": ("application/zip", "zip"),
}


def export(result: ScanResult, fmt: str) -> tuple[bytes, str, str]:
    """(bytes, media type, file extension) of ``result`` in ``fmt`` (json | csv | wfdb)."""
    if fmt not in EXPORTS:
        raise ValueError(f"format must be one of {sorted(EXPORTS)}")
    media, ext = EXPORTS[fmt]
    if fmt == "json":
        import json

        return json.dumps(result.to_json_dict(), ensure_ascii=False).encode(), media, ext
    if fmt == "csv":
        return to_csv(result).encode(), media, ext
    return to_wfdb_zip(result), media, ext
