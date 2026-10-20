"""What a record must be before it is measured (D97).

Every input route (files, API uploads, DICOM network, digitised pages) ends in
``run_pipeline``, which calls :func:`check_record` first. A record that is not
measurable is refused with the reason — never measured on guesses (TZ §0):

* no leads or no samples, a sampling rate below ``min_sampling_rate_hz`` (50 Hz gave
  QRS 178 ms and false LBBB / short QT), or less than ``min_duration_s`` of signal;
* invalid samples (NaN / inf: WFDB marks missing samples so, a CSV may hold "nan").
  One NaN used to spread through the band-pass over the whole lead and silently
  remove every measurement. Runs at the record's start or end are trimmed; inside,
  runs up to ``max_gap_s`` are bridged linearly and listed in
  ``provenance.sample_mask`` (as for digitised gaps); a longer run, a lead with no
  valid sample, or more than ``max_invalid_fraction`` invalid overall refuses.

A sampling rate below ``low_sampling_rate_hz`` is measured with a note.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation


def _runs(bad: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    d = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def check_record(sig: SignalRepresentation, limits) -> SignalRepresentation:
    """``sig`` if it is measurable as is, a repaired copy (trimmed / bridged, noted), or
    ``ValueError`` with the reason. ``limits``: ``cardiolens.config.IngestionConfig``."""
    fs = float(sig.sampling_rate_hz)
    if sig.n_leads == 0 or sig.n_samples == 0:
        raise ValueError("record has no leads or no samples")
    if fs < limits.min_sampling_rate_hz:
        raise ValueError(f"sampling rate {fs:g} Hz is below {limits.min_sampling_rate_hz:g} Hz: "
                         "QRS and intervals cannot be measured")
    x = np.asarray(sig.signal)
    bad = ~np.isfinite(x)
    notes: list[str] = []
    out = sig
    if bad.any():
        col_bad = bad.all(axis=0)
        good_cols = np.flatnonzero(~col_bad)
        if good_cols.size == 0:
            raise ValueError("record has no valid samples")
        a, b = int(good_cols[0]), int(good_cols[-1]) + 1
        if a or b < x.shape[1]:
            notes.append(f"{(a + x.shape[1] - b) / fs:.2f} s of invalid samples trimmed at the record's ends")
        x, bad = x[:, a:b].astype(np.float32, copy=True), bad[:, a:b]
        dead = [n for n, row in zip(sig.lead_names, bad, strict=True) if row.all()]
        if dead:
            raise ValueError(f"lead {', '.join(dead)} has no valid samples")
        if bad.mean() > limits.max_invalid_fraction:
            raise ValueError(f"{100 * bad.mean():.1f}% of the samples are invalid (NaN/inf)")
        mask: dict[str, list[tuple[float, float]]] = dict(sig.provenance.sample_mask or {})
        t = np.arange(x.shape[1])
        for i, name in enumerate(sig.lead_names):
            if not bad[i].any():
                continue
            for r0, r1 in _runs(bad[i]):
                if (r1 - r0) / fs > limits.max_gap_s:
                    raise ValueError(f"lead {name}: {(r1 - r0) / fs:.2f} s of invalid samples at "
                                     f"{r0 / fs:.2f} s (more than {limits.max_gap_s:g} s cannot be bridged)")
                mask.setdefault(name, []).append((r0 / fs, r1 / fs))
            x[i] = np.interp(t, t[~bad[i]], x[i, ~bad[i]])
        bridged = sum(len(v) for v in mask.values()) - sum(len(v) for v in (sig.provenance.sample_mask or {}).values())
        if bridged:
            notes.append(f"{bridged} short run(s) of invalid samples bridged and masked")
        prov = sig.provenance.model_copy(update={"sample_mask": mask or None})
        out = sig.model_copy(update={"signal": x, "duration_s": x.shape[1] / fs, "provenance": prov})
    if out.n_samples / fs < limits.min_duration_s:
        raise ValueError(f"record is {out.n_samples / fs:.2f} s long; at least {limits.min_duration_s:g} s "
                         "is needed to measure")
    if fs < limits.low_sampling_rate_hz:
        notes.append(f"sampling rate {fs:g} Hz: QRS edges are coarse (< {limits.low_sampling_rate_hz:g} Hz)")
    if notes:
        prov = out.provenance.model_copy(update={"notes": [*out.provenance.notes, *notes]})
        out = out.model_copy(update={"provenance": prov})
    return out
