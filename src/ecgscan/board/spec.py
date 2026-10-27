"""Board geometry from ``configs/scanner/boards/<name>.yaml``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgscan.config import load_yaml


@dataclass(frozen=True)
class BoardSpec:
    name: str
    title: str
    size_mm: tuple[float, float]
    dictionary: str
    marker_mm: float
    markers: dict[int, tuple[float, float]]
    zone_mm: tuple[float, float, float, float]
    ruler_mm: float = 6.0          # longest ruler tick (10 mm marks); shorter on a tight board (D176)

    def corners_mm(self, marker_id: int) -> np.ndarray:
        """The marker's 4 corners in board mm, in ArUco order (TL, TR, BR, BL)."""
        cx, cy = self.markers[marker_id]
        h = self.marker_mm / 2
        return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], np.float64)

    @property
    def zone_size_mm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.zone_mm
        return x1 - x0, y1 - y0


def load_board(name: str) -> BoardSpec:
    d = load_yaml(f"boards/{name}")
    spec = BoardSpec(name=d["name"], title=d.get("title", d["name"]), size_mm=tuple(d["size_mm"]),
                     dictionary=d["dictionary"], marker_mm=float(d["marker_mm"]),
                     markers={int(k): (float(v[0]), float(v[1])) for k, v in d["markers"].items()},
                     zone_mm=tuple(float(v) for v in d["zone_mm"]),
                     ruler_mm=float(d.get("ruler_mm", 6.0)))
    x0, y0, x1, y1 = spec.zone_mm
    if not (0 <= x0 < x1 <= spec.size_mm[0] and 0 <= y0 < y1 <= spec.size_mm[1]):
        raise ValueError(f"board {name}: zone outside the board")
    return spec


def load_boards(names: list[str]) -> list[BoardSpec]:
    specs = [load_board(n) for n in names]
    seen: dict[int, str] = {}
    for s in specs:
        for i in s.markers:
            if i in seen:
                raise ValueError(f"marker id {i} is on both {seen[i]} and {s.name}: a photo could not tell them apart")
            seen[i] = s.name
    return specs
