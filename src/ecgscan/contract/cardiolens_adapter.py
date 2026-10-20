"""``ScanResult`` -> CardioLens ``SignalRepresentation`` (REPO_AUDIT §9, SD-006).

``cardiolens`` is imported lazily, so ``ecgscan`` stays usable without it. Mapping:

* mV -> µV; every trace is placed on one time axis at ``sampling_rate_hz``;
* outside a lead's time window the samples are 0 and ``provenance.lead_windows``
  says where the lead has data (what Layer 3 already understands);
* a rhythm strip of a lead replaces its panel and widens its window — unless it is
  flagged ``RHYTHM_MISMATCH`` (it disagrees with every panel recorded with it, SD-060);
* missing samples inside a window are linearly bridged **on the CardioLens side**
  only, so the filters can run; where they were goes to ``provenance.sample_mask``
  and Layer 3 measures no beat on them (M4). The ``ScanResult`` keeps them missing.
"""

from __future__ import annotations

import numpy as np

from ecgscan.contract.models import DIGITIZED_FLAG, RHYTHM_MISMATCH, STANDARD_LEADS, ScanResult


class RejectedScanError(ValueError):
    """A rejected scan has no signal to hand over."""

    def __init__(self, reasons: list[str]):
        super().__init__(f"scan rejected: {', '.join(reasons)}")
        self.reasons = reasons


def _bridge_gaps(x: np.ndarray) -> tuple[np.ndarray, int]:
    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, 0
    if n_bad == len(x):
        return np.zeros_like(x), n_bad
    idx = np.arange(len(x))
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out, n_bad


def _gap_intervals(missing: np.ndarray, fs: float) -> dict[str, list[tuple[float, float]]]:
    """Per lead, the [t0, t1) seconds of bridged samples (SD-006: a mask, not data)."""
    out = {}
    for row, lead in enumerate(STANDARD_LEADS):
        m = missing[row]
        if m.any():
            d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
            out[lead] = [(round(a / fs, 4), round(b / fs, 4))
                         for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)]
    return out


def to_signal_representation(result: ScanResult):
    """Build the CardioLens input signal; raises ``RejectedScanError`` when rejected."""
    from cardiolens.contracts.signal import Provenance, SignalRepresentation

    if result.status == "rejected":
        raise RejectedScanError(list(result.reject_reasons))
    fs = float(result.sampling_rate_hz)
    traces = result.all_traces()
    if not traces:
        raise ValueError("scan result holds no traces")
    end_s = max(t.start_s + len(t.samples_mv) / fs for _, t in traces)
    n = int(round(end_s * fs))
    signal = np.zeros((len(STANDARD_LEADS), n), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    bridged: dict[str, int] = {}
    missing = np.zeros_like(signal, dtype=bool)
    # Panels first, rhythm strips after: a strip overwrites its lead's panel.
    # A strip flagged RHYTHM_MISMATCH (unrelated to its panels, SD-060) is not measured.
    strips = [(t.lead, t) for t in result.rhythm if RHYTHM_MISMATCH not in t.flags]
    for lead, t in [(k, v) for k, v in result.leads.items()] + strips:
        if lead not in STANDARD_LEADS:
            continue
        raw = t.array()
        arr, n_bad = _bridge_gaps(raw)
        i0 = int(round(t.start_s * fs))
        seg = arr[: max(0, n - i0)]
        row = STANDARD_LEADS.index(lead)
        signal[row, i0:i0 + len(seg)] = (seg * 1000.0).astype(np.float32)
        missing[row, i0:i0 + len(seg)] = ~np.isfinite(raw[: len(seg)])
        windows[lead] = (round(i0 / fs, 6), round((i0 + len(seg)) / fs, 6))
        if n_bad:
            bridged[lead] = n_bad
    cal = result.calibration
    notes = [
        f"ecgscan scan_id={result.scan_id} status={result.status} pipeline={result.source.pipeline_version}",
        (f"calibration: speed={cal.paper_speed_mm_s} ({cal.speed_source}) gain={cal.gain_mm_mv} "
        f"({cal.gain_source}) px/mm={cal.px_per_mm_x} conf={cal.confidence}"),
        f"layout: {result.layout.name} conf={result.layout.confidence}",
    ]
    if result.warnings:
        notes.append("warnings: " + ", ".join(result.warnings))
    if bridged:
        notes.append("missing samples bridged on the CardioLens side: "
                     + ", ".join(f"{k}={v}" for k, v in sorted(bridged.items())))
    if DIGITIZED_FLAG in result.provenance_flags:
        notes.append(DIGITIZED_FLAG)
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_LEADS), duration_s=n / fs,
        source="image",
        provenance=Provenance(digitized=True, confidence=round(float(result.quality.overall), 3),
                              lead_windows=windows, sample_mask=_gap_intervals(missing, fs) or None, notes=notes),
    )
