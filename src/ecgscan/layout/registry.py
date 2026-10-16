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
    pulse = PulseSpec(**(d.get("pulse") or {}))
    return LayoutSpec(
        name=d["name"], rows=tuple(tuple(r) for r in d["rows"]), panel_s=float(d["panel_s"]),
        rhythm=tuple(d.get("rhythm") or ()), row_pitch_mm=float(d.get("row_pitch_mm", 32.0)),
        header_mm=float(d.get("header_mm", 20.0)), margin_mm=float(d.get("margin_mm", 8.0)),
        pulse=pulse, label_offset_mm=tuple(d.get("label_offset_mm", (2.0, 4.0))),
        registry_version=str(d.get("registry_version", "1")), confidence=str(d.get("confidence", "medium")),
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
