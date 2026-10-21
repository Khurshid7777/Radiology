"""Layout registry: page formats as YAML in ``configs/scanner/layouts`` (TZ §5.6, SD-017).

A new machine format is a new file, not a code change. The registry describes the
*time* structure (which lead is printed where and when) and the nominal geometry
in millimetres; the renderer and, from M3 on, the layout detector both use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ecgscan.config import config_dir, load_yaml


@dataclass(frozen=True)
class PulseSpec:
    present: bool = True
    width_s: float = 0.2
    area_mm: float = 8.0
    every_n_rows: int = 1
    rows: tuple[int, ...] | None = None    # drawn on these rows only (a strip's single start pulse)


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    rows: tuple[tuple[str, ...], ...]
    panel_s: float
    rhythm: tuple[str, ...] = ()
    row_pitch_mm: float = 32.0
    header_mm: float = 20.0
    margin_mm: float = 8.0
    pulse: PulseSpec = field(default_factory=PulseSpec)
    label_offset_mm: tuple[float, float] = (2.0, 4.0)
    registry_version: str = "1"
    confidence: str = "medium"
    # Blank paper between consecutive columns (3-channel strip recorders leave ~10 mm and
    # print the next group's labels there). 0 for pages whose columns abut (3x4, 6x2).
    panel_gap_mm: float = 0.0
    # A strip prints its pulse once, at the start: a photo without it does not show the
    # first column, and naming its groups by position would mislabel every lead (SD-059).
    start_pulse_required: bool = False

    def column_offset_mm(self, c: int, speed_mm_s: float) -> float:
        """Where column ``c`` starts, in mm from the first column's start."""
        return c * (self.panel_s * speed_mm_s + self.panel_gap_mm)

    def width_mm(self, speed_mm_s: float) -> float:
        """Paper length of the panel columns (with the gaps between them)."""
        return self.column_offset_mm(self.n_columns - 1, speed_mm_s) + self.panel_s * speed_mm_s

    @property
    def n_columns(self) -> int:
        return max(len(r) for r in self.rows)

    @property
    def record_s(self) -> float:
        """Length of the underlying record shown on the page."""
        return self.panel_s * self.n_columns

    @property
    def n_trace_rows(self) -> int:
        return len(self.rows) + len(self.rhythm)

    def lead_windows(self) -> dict[str, tuple[float, float]]:
        """Panel lead -> (t0, t1) in seconds of the record."""
        out: dict[str, tuple[float, float]] = {}
        for row in self.rows:
            for c, lead in enumerate(row):
                out[lead] = (c * self.panel_s, (c + 1) * self.panel_s)
        return out

    def leads(self) -> list[str]:
        return [lead for row in self.rows for lead in row]


def _from_dict(d: dict) -> LayoutSpec:
    pd = dict(d.get("pulse") or {})
    if pd.get("rows") is not None:
        pd["rows"] = tuple(int(r) for r in pd["rows"])
    pulse = PulseSpec(**pd)
    return LayoutSpec(
        name=d["name"], rows=tuple(tuple(r) for r in d["rows"]), panel_s=float(d["panel_s"]),
        rhythm=tuple(d.get("rhythm") or ()), row_pitch_mm=float(d.get("row_pitch_mm", 32.0)),
        header_mm=float(d.get("header_mm", 20.0)), margin_mm=float(d.get("margin_mm", 8.0)),
        pulse=pulse, label_offset_mm=tuple(d.get("label_offset_mm", (2.0, 4.0))),
        registry_version=str(d.get("registry_version", "1")), confidence=str(d.get("confidence", "medium")),
        panel_gap_mm=float(d.get("panel_gap_mm", 0.0)), start_pulse_required=bool(d.get("start_pulse_required", False)),
    )


def available_layouts() -> dict[str, LayoutSpec]:
    out = {}
    for p in sorted((config_dir() / "layouts").glob("*.yaml")):
        spec = _from_dict(load_yaml(f"layouts/{p.name}"))
        if spec.name in out:
            raise ValueError(f"duplicate layout name {spec.name!r} in {p.name}")
        out[spec.name] = spec
    return out


def get_layout(name: str) -> LayoutSpec:
    layouts = available_layouts()
    if name not in layouts:
        raise KeyError(f"unknown layout {name!r}; known: {sorted(layouts)}")
    return layouts[name]
