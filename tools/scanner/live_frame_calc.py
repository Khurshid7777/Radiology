"""SD-156: the numbers behind the live capture frame ("green frame"): how close the phone must be for the scanner's
8 px/mm, what a camera's resolution allows to fit in one shot, and how much a phone held at an angle distorts the time
scale. Pinhole camera, the sheet flat; no photo is read.

    .venv/bin/python tools/scanner/live_frame_calc.py            # the tables of docs/scanner/LIVE_FRAME.md
"""

from __future__ import annotations

import argparse
import math

HFOV_DEG = 67.0             # a phone's main camera, 26 mm equivalent, long side of a 4:3 frame (16:9 video is cut from it)
MIN_PX_PER_MM = 8.0         # configs/scanner/capture_qc.yaml thresholds.min_px_per_mm
FILL = 0.9                  # the object takes this share of the frame's long side (room for the hand)
WIDTHS = {"видео 1280×720": 1280, "видео 1920×1080": 1920, "видео 2560×1440": 2560, "видео 3840×2160": 3840,
          "снимок 12 Мп (4000×3000)": 4000}
# (long side, short side), mm. The 3-channel strip's paper width is taken as 80 mm until the owner's strip is measured.
OBJECTS = {"лист A4 поперёк (297 мм)": (297.0, 210.0), "лента 3 канала, 10 с (250 мм)": (250.0, 80.0),
           "часть ленты, 6 с (150 мм)": (150.0, 80.0)}
MAX_SPREAD = 0.08           # configs/scanner/capture_qc.yaml live.max_scale_spread


def visible_mm(distance_mm: float, hfov_deg: float = HFOV_DEG) -> float:
    """The width of the sheet's plane the frame's long side sees at this distance."""
    return 2.0 * distance_mm * math.tan(math.radians(hfov_deg) / 2.0)


def distance_mm(visible: float, hfov_deg: float = HFOV_DEG) -> float:
    return visible / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def px_per_mm(width_px: int, visible: float) -> float:
    return width_px / visible


def max_object_mm(width_px: int, min_ppm: float = MIN_PX_PER_MM, fill: float = FILL) -> float:
    """The longest object one shot of this width takes at ``min_ppm`` with the object filling ``fill`` of the frame."""
    return fill * width_px / min_ppm


def shot_for(object_mm: float, width_px: int, *, fill: float = FILL, hfov_deg: float = HFOV_DEG) -> dict:
    """The object framed to ``fill``: the distance, the px/mm it gets and whether that is enough."""
    vis = object_mm / fill
    ppm = px_per_mm(width_px, vis)
    return {"distance_cm": distance_mm(vis, hfov_deg) / 10.0, "px_per_mm": ppm, "enough": ppm >= MIN_PX_PER_MM,
            "parts": max(1, math.ceil(object_mm / max_object_mm(width_px, fill=fill)))}


def yaw_spread(deg: float, object_mm: float, distance: float) -> float:
    """The phone turned about the vertical by ``deg``: the 5-mm period of the vertical lines in the sheet's left third
    over the right third, minus 1. The spacing of the lines goes with the square of the magnification (the plane is both
    nearer and foreshortened), so this is twice the pitch case (checked on warped stand sheets, SD-156)."""
    s = (object_mm / 3.0) * math.sin(math.radians(deg))
    return ((distance + s) / (distance - s)) ** 2 - 1.0


def pitch_spread(deg: float, object_mm: float, distance: float) -> float:
    """The phone turned about the horizontal: the vertical lines' period in the top third over the bottom third, minus 1."""
    s = (object_mm / 3.0) * math.sin(math.radians(deg))
    return (distance + s) / (distance - s) - 1.0


def angle_for(spread: float, object_mm: float, distance: float, *, yaw: bool = True) -> float:
    """The tilt (degrees) at which the spread reaches ``spread``, by bisection."""
    f = yaw_spread if yaw else pitch_spread
    lo, hi = 0.0, 45.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if f(mid, object_mm, distance) < spread else (lo, mid)
    return (lo + hi) / 2.0


def tables(max_spread: float = MAX_SPREAD) -> str:
    out = [(f"Камера: горизонтальный угол обзора {HFOV_DEG:.0f}°, объект на {FILL:.0%} длинной стороны кадра, "
            f"нужно {MIN_PX_PER_MM:.0f} точек на мм."), "",
           "| Разрешение | Самый длинный объект за 1 снимок | " + " | ".join(OBJECTS) + " |",
           "|---|---|" + "---|" * len(OBJECTS)]
    for name, wpx in WIDTHS.items():
        cells = []
        for obj, _ in OBJECTS.values():
            s = shot_for(obj, wpx)
            more = "" if s["enough"] else f", мало → {s['parts']} части"
            cells.append(f"{s['px_per_mm']:.1f} т/мм, {round(s['distance_cm'])} см{more}")
        out.append(f"| {name} | {max_object_mm(wpx):.0f} мм | " + " | ".join(cells) + " |")
    out += ["", f"Наклон телефона, при котором разброс шага сетки по третям листа достигает {max_spread:.0%}:", "",
            "| Объект | Расстояние | Поворот влево-вправо | Наклон вверх-вниз |", "|---|---|---|---|"]
    for name, (obj, side) in OBJECTS.items():
        d = distance_mm(obj / FILL)
        out.append(f"| {name} | {d / 10:.0f} см | {angle_for(max_spread, obj, d):.1f}° | "
                   f"{angle_for(max_spread, side, d, yaw=False):.1f}° |")
    d = distance_mm(297.0 / FILL)
    out += ["", "Ошибка шкалы времени по краям листа A4 относительно середины при повороте влево-вправо:", "",
            "| Поворот | Разброс по третям | Край против середины |", "|---|---|---|"]
    for deg in (1, 2, 3, 5, 10):
        edge = ((d + 148.5 * math.sin(math.radians(deg))) / d) ** 2 - 1.0
        out.append(f"| {deg}° | {yaw_spread(deg, 297.0, d):.1%} | ±{edge:.1%} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-spread", type=float, default=MAX_SPREAD)
    print(tables(ap.parse_args().max_spread))


if __name__ == "__main__":
    main()
