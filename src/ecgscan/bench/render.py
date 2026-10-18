"""Synthetic paper-ECG pages with exact ground truth (TZ §11, M1).

Draws any registry layout (``configs/scanner/layouts``) at any paper speed, gain and
resolution — the knobs the legacy renderer (``cardiolens.layer1_ingestion.digitize.
render_ecg_page``) fixes: it is 3x4-only, integer px/mm and has no printed header.
Everything is drawn with sub-pixel anti-aliased lines (``cv2`` shift bits), so a
fractional px/mm is honoured exactly.

The ground truth of a page is the signal segment behind every drawn trace (in mV,
on the bench time axis at ``fs``) plus the page geometry in pixels; ``ecgscan.bench.
degrade`` adds the photo defects and the transform from page to photo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ecgscan.layout.registry import LayoutSpec

_SHIFT = 4                      # cv2 fixed-point bits: coordinates in 1/16 px
_ONE = 1 << _SHIFT

# Grid palettes (BGR): minor line, major line. Thermal papers differ by vendor.
GRID_PALETTES = {
    "red": ((205, 205, 250), (150, 150, 235)),
    "pink": ((215, 200, 245), (175, 150, 230)),
    "orange": ((190, 215, 250), (120, 175, 240)),
    "green": ((205, 235, 205), (150, 205, 150)),
    "gray": ((220, 220, 220), (175, 175, 175)),
}
PAPER_BGR = (247, 249, 250)


@dataclass
class RenderSpec:
    px_per_mm: float = 12.0
    speed_mm_s: float = 25.0
    gain_mm_mv: float = 10.0
    grid: str = "red"
    trace_mm: float = 0.25          # printed stroke width
    trace_gray: int = 25            # 0 = black; thermal fading raises it
    grid_fade: float = 0.0          # 0..1: grid blends towards paper
    header: bool = True             # printed patient/settings header
    header_settings: bool = True    # "25 mm/s 10 mm/mV" in the header (OCR source, M2)
    labels: bool = True
    row_pitch_mm: float | None = None   # override the layout's baseline pitch
    column_markers: bool = False    # short ticks at the column boundaries
    text_gray: int | None = None    # header and labels; None = the pen (M6 masks: erase the pen, keep the text)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceGT:
    lead: str
    kind: str                       # "panel" | "rhythm"
    start_s: float
    samples_mv: np.ndarray          # at PageGT.fs
    x0_px: float                    # page coordinates of the trace origin
    baseline_px: float


@dataclass
class PageGT:
    layout: str
    speed_mm_s: float
    gain_mm_mv: float
    px_per_mm: float
    fs: float
    width_px: int
    height_px: int
    header_box_px: tuple[int, int, int, int] | None
    traces: list[TraceGT] = field(default_factory=list)

    def meta(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "traces"}
        d["traces"] = [{"lead": t.lead, "kind": t.kind, "start_s": t.start_s, "n": len(t.samples_mv),
                        "x0_px": round(t.x0_px, 2), "baseline_px": round(t.baseline_px, 2)} for t in self.traces]
        return d


def _fx(v: float) -> int:
    return int(round(v * _ONE))


def _blend(color: tuple[int, int, int], fade: float) -> tuple[int, int, int]:
    return tuple(int(round(c + (p - c) * fade)) for c, p in zip(color, PAPER_BGR, strict=True))  # type: ignore[return-value]


def _draw_grid(img: np.ndarray, ppm: float, palette: str, fade: float) -> None:
    import cv2

    minor, major = (_blend(c, fade) for c in GRID_PALETTES[palette])
    h, w = img.shape[:2]
    major_th = max(1, int(round(0.12 * ppm)))
    for k in range(int(w / ppm) + 1):
        x = _fx(k * ppm)
        is_major = k % 5 == 0
        cv2.line(img, (x, 0), (x, _fx(h)), major if is_major else minor, major_th if is_major else 1,
                 cv2.LINE_AA, _SHIFT)
    for k in range(int(h / ppm) + 1):
        y = _fx(k * ppm)
        is_major = k % 5 == 0
        cv2.line(img, (0, y), (_fx(w), y), major if is_major else minor, major_th if is_major else 1,
                 cv2.LINE_AA, _SHIFT)


def _polyline(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, color, thickness: int) -> None:
    import cv2

    pts = np.stack([np.round(xs * _ONE), np.round(ys * _ONE)], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA,
                  shift=_SHIFT)


def _text(img: np.ndarray, s: str, x: float, y: float, height_px: float, color) -> None:
    import cv2

    scale = max(0.2, height_px / 22.0)
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                max(1, int(round(scale * 1.5))), cv2.LINE_AA)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    return np.interp(np.arange(n_out) / fs_out, t_in, x)


def render_page(signals_mv: dict[str, np.ndarray], fs_in: float, layout: LayoutSpec, spec: RenderSpec,
                *, fs: float = 500.0) -> tuple[np.ndarray, PageGT]:
    """Draw ``signals_mv`` (lead -> mV at ``fs_in``) as a paper page; return (BGR image, truth)."""
    ppm, speed, gain = float(spec.px_per_mm), float(spec.speed_mm_s), float(spec.gain_mm_mv)
    sig = {k: resample_to(v, fs_in, fs) for k, v in signals_mv.items()}
    need = int(round(layout.record_s * fs))
    for lead in set(layout.leads()) | set(layout.rhythm):
        if lead not in sig or len(sig[lead]) < need:
            raise ValueError(f"lead {lead} needs {layout.record_s} s of signal")
    pitch = spec.row_pitch_mm or layout.row_pitch_mm
    pulse_area = layout.pulse.area_mm if layout.pulse.present else 0.0
    header = layout.header_mm if spec.header else 0.0
    trace_x0_mm = layout.margin_mm + pulse_area
    width_mm = trace_x0_mm + layout.record_s * speed + layout.margin_mm
    n_rows = layout.n_trace_rows
    height_mm = header + 2 * layout.margin_mm + pitch * n_rows
    w, h = int(np.ceil(width_mm * ppm)), int(np.ceil(height_mm * ppm))
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = PAPER_BGR
    _draw_grid(img, ppm, spec.grid, spec.grid_fade)
    ink = (spec.trace_gray,) * 3
    text_ink = ink if spec.text_gray is None else (spec.text_gray,) * 3
    thick = max(1, int(round(spec.trace_mm * ppm)))
    label_h = 2.5 * ppm
    gt = PageGT(layout=layout.name, speed_mm_s=speed, gain_mm_mv=gain, px_per_mm=ppm, fs=fs,
                width_px=w, height_px=h, header_box_px=None)
    if spec.header:
        y0, y1 = int(layout.margin_mm * 0.5 * ppm), int((layout.margin_mm * 0.5 + header) * ppm)
        gt.header_box_px = (int(layout.margin_mm * ppm), y0, int(min(w, 140 * ppm)), y1)
        _text(img, "Name: TEST PATIENT   ID: 0000000   Age: 45   Sex: M", layout.margin_mm * ppm,
              y0 + 5 * ppm, 3.0 * ppm, text_ink)
        if spec.header_settings:
            _text(img, f"{speed:g} mm/s   {gain:g} mm/mV   0.05-150 Hz   50 Hz", layout.margin_mm * ppm,
                  y0 + 11 * ppm, 3.0 * ppm, text_ink)

    def baseline(r: int) -> float:
        return (header + layout.margin_mm + pitch * (r + 0.5)) * ppm

    def draw_trace(lead: str, kind: str, t0: float, t1: float, row: int) -> None:
        i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
        seg = sig[lead][i0:i1]
        x0 = (trace_x0_mm + t0 * speed) * ppm
        xs = x0 + np.arange(len(seg)) / fs * speed * ppm
        ys = baseline(row) - seg * gain * ppm
        _polyline(img, xs, ys, ink, thick)
        gt.traces.append(TraceGT(lead=lead, kind=kind, start_s=t0, samples_mv=seg.copy(), x0_px=x0,
                                 baseline_px=baseline(row)))
        if spec.labels:
            dx, dy = layout.label_offset_mm
            _text(img, lead, x0 + dx * ppm, baseline(row) - dy * ppm, label_h, text_ink)

    for r, row in enumerate(layout.rows):
        for c, lead in enumerate(row):
            draw_trace(lead, "panel", c * layout.panel_s, (c + 1) * layout.panel_s, r)
        if spec.column_markers:
            for c in range(1, len(row)):
                x = (trace_x0_mm + c * layout.panel_s * speed) * ppm
                _polyline(img, np.array([x, x]), np.array([baseline(r) - 2 * ppm, baseline(r) + 2 * ppm]), ink, thick)
    for k, lead in enumerate(layout.rhythm):
        draw_trace(lead, "rhythm", 0.0, layout.record_s, len(layout.rows) + k)
    if layout.pulse.present:
        pw = min(layout.pulse.width_s * speed, pulse_area - 2.0) * ppm
        ph = 1.0 * gain * ppm
        for r in range(n_rows):
            if r % max(1, layout.pulse.every_n_rows):
                continue
            xa, yb = (layout.margin_mm + 1.0) * ppm, baseline(r)
            _polyline(img, np.array([xa - 1.0 * ppm, xa, xa, xa + pw, xa + pw, xa + pw + 1.0 * ppm]),
                      np.array([yb, yb, yb - ph, yb - ph, yb, yb]), ink, thick)
    return img, gt
