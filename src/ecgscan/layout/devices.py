"""Device profiles: what a recording machine prints and where (TZ 18.5, S2).

``configs/scanner/devices/<id>.yaml``; files starting with ``_`` are examples, not loaded.
The registry starts empty and is filled from real machines only. A profile is a prior: when
the page's own rows leave the layout undecided, the profile's single layout is taken and the
doctor still confirms (``LAYOUT_UNCERTAIN`` stays); it never overrides what the page shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ecgscan.config import config_dir, load_yaml


@dataclass(frozen=True)
class DeviceProfile:
    id: str
    manufacturer: str | None = None
    model: str | None = None
    layouts: tuple[str, ...] = ()
    header: str | None = None
    pulse: str | None = None
    speeds_mm_s: tuple[float, ...] = ()
    gains_mm_mv: tuple[float, ...] = ()
    notes: str | None = None
    extra: dict = field(default_factory=dict)


def _profile(pid: str, d: dict) -> DeviceProfile:
    known = {"manufacturer", "model", "layouts", "header", "pulse", "speeds_mm_s", "gains_mm_mv", "notes"}
    return DeviceProfile(id=pid, manufacturer=d.get("manufacturer"), model=d.get("model"),
                         layouts=tuple(d.get("layouts") or ()), header=d.get("header"), pulse=d.get("pulse"),
                         speeds_mm_s=tuple(float(v) for v in d.get("speeds_mm_s") or ()),
                         gains_mm_mv=tuple(float(v) for v in d.get("gains_mm_mv") or ()),
                         notes=d.get("notes"), extra={k: v for k, v in d.items() if k not in known})


def device_profiles() -> dict[str, DeviceProfile]:
    """Every loaded profile by id (its file name); examples (``_*.yaml``) are skipped."""
    root = config_dir() / "devices"
    if not root.is_dir():
        return {}
    return {p.stem: _profile(p.stem, load_yaml(f"devices/{p.name}"))
            for p in sorted(root.glob("*.yaml")) if not p.name.startswith("_")}


def device_profile(pid: str | None) -> DeviceProfile | None:
    return device_profiles().get(pid) if pid else None
