"""Annotated ECG waveform rendering (TZ §11.3).

Draws the classic ECG grid (small 1 mm / large 5 mm boxes, 25 mm/s, 10 mm/mV),
plots selected leads, and overlays R-peak markers and finding labels. Renders to
SVG (vector, for the report) or PNG.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cardiolens.pipeline import PipelineResult

# Standard scale: 25 mm/s, 10 mm/mV. One large box = 0.2 s x 0.5 mV.
_MM_PER_S = 25.0
_MM_PER_MV = 10.0


MAX_LEADS = 4


def short_label(label: str) -> str:
    """'Элевация ST, нижняя территория (II, III, aVF)' -> 'Элевация ST'."""
    return label.split(" (")[0].split(" в отведениях")[0].split(",")[0].split(" — ")[0].strip()


def finding_leads(result: PipelineResult) -> tuple[list[str], dict[str, str]]:
    """The leads to draw (D196): those of the leading finding first — a critical one, then
    ischaemia, then any other that names its leads — up to four, each with the finding's name.
    None named: ([], {}), and the default I, II, V1, V5 is drawn."""
    present = set(result.signal.lead_names)
    found = [f for f in result.findings.findings if f.evidence_map().get("leads")]
    found.sort(key=lambda f: (not f.critical, f.group != "ischemia"))
    leads: list[str] = []
    marked: dict[str, str] = {}
    for f in found:
        for ld in f.evidence_map()["leads"].split(","):
            ld = ld.strip()
            if ld in present and ld not in marked and len(leads) < MAX_LEADS:
                leads.append(ld)
                marked[ld] = short_label(f.label_ru)
    return leads, marked


def render_waveform(
    result: PipelineResult,
    path: str,
    *,
    leads: list[str] | None = None,
    seconds: float = 5.0,
) -> str:
    sig = result.signal
    fs = sig.sampling_rate_hz
    marked: dict[str, str] = {}
    if leads is None:
        leads, marked = finding_leads(result)       # D196: the leads the findings name
    leads = leads or [ln for ln in ("I", "II", "V1", "V5") if ln in sig.lead_names]
    total = sig.n_samples / fs
    # A digitised 3x4 page holds each lead only inside its 2.5-s slot; outside it the samples
    # are bridged, not data (Provenance.lead_windows). Each lead is drawn in its own slot,
    # on the record's time axis, so V1 or V5 is not a flat line at 0-5 s (S4 review, D184).
    windows = sig.provenance.lead_windows or {}

    fig, axes = plt.subplots(len(leads), 1, figsize=(10, 2.0 * len(leads)))
    if len(leads) == 1:
        axes = [axes]

    for ax, lead in zip(axes, leads):
        t0, t1 = windows.get(lead, (0.0, total))
        t1 = min(t1, t0 + seconds, total)
        i0, i1 = int(t0 * fs), max(int(t0 * fs) + 1, int(t1 * fs))
        t = [i / fs for i in range(i0, i1)]
        row_mv = sig.lead(lead)[i0:i1] / 1000.0  # µV -> mV for display
        ax.plot(t, row_mv, color="#b00020", linewidth=0.8)
        # ECG grid.
        ax.grid(which="major", color="#f4a6a6", linewidth=0.6)
        ax.grid(which="minor", color="#fbe0e0", linewidth=0.3)
        ax.set_xlim(t0, t0 + seconds)
        ax.set_xticks([t0 + x * 0.2 for x in range(int(seconds / 0.2) + 1)])
        ax.set_xticklabels([f"{t0 + x * 0.2:.1f}" if x % 5 == 0 else "" for x in range(int(seconds / 0.2) + 1)],
                           fontsize=7)
        ax.minorticks_on()
        ax.set_ylabel(f"{lead}\n(mV)", rotation=0, ha="right", va="center", fontsize=8)
        if lead in marked:        # D196: highlighted and labelled with its finding
            ax.set_facecolor("#fff4cc")
            ax.text(0.995, 0.92, marked[lead], transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#7a4f00", bbox={"boxstyle": "round,pad=0.2", "fc": "#ffe9a8", "ec": "none"})
        # R-peak markers.
        for rp in result.measurements.r_peaks_samples:
            if i0 <= rp < i1:
                ax.axvline(rp / fs, color="#1565c0", alpha=0.25, linewidth=0.6)

    axes[-1].set_xlabel("Время, с (25 мм/с, 10 мм/мВ)", fontsize=8)
    names = [short_label(f.label_ru) for f in result.findings.findings]
    finding_labels = "; ".join(dict.fromkeys(names)) or "—"
    if len(finding_labels) > 110:
        finding_labels = finding_labels[:107] + "…"
    fig.suptitle(f"ЭКГ — находки: {finding_labels}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, format="svg" if path.endswith(".svg") else None, dpi=120)
    plt.close(fig)
    return path
