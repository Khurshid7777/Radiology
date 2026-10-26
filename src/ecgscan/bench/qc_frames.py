"""Known-good and known-bad camera frames for the frame checks (TZ addendum §18.14, S1).

Acceptance of S1: the checks refuse >= 90 % of the bad frames and <= 10 % of the good
ones. Frames are 1280-px previews of a synthetic page photographed on a desk (the size
the app checks); ``scale_to_capture`` maps them to the photo the phone would take.

Every frame is drawn from a seed; one class = one defect, everything else as in the
good class. Classes and their parameters are in ``CLASSES``; what counts as bad is the
TZ table §18.2 (blur, glare, too dark / too light, shadow, too far, sheet cut by the
frame, rotated past the limit). Camera tilt out of the sheet's plane is checked by the
gyroscope on the phone and cannot be judged from a still: not in this set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ecgscan.bench.degrade import DegradeSpec, degrade

FRAME_W = 1280
PAGE_PPM = 5.0                      # the page is rendered at this scale before the desk and the camera

GOOD = dict(background=(0.08, 0.16), perspective=(0.0, 0.02), rotation_deg=(0.0, 2.5), noise=(1.0, 3.0),
            vignette=(0.0, 0.15), shadow=(0.0, 0.12), blur_mm=(0.0, 0.08), exposure=(1.0, 1.0),
            photo_w=(3600, 4300))

# One defect per class: the keys replace the good ranges.
CLASSES: dict[str, dict] = {
    "good": {},
    "blur": dict(blur_mm=(0.45, 0.8)),
    "glare": dict(glare=(0.7, 1.0)),
    "dark": dict(exposure=(0.25, 0.38)),
    "bright": dict(exposure=(1.45, 1.8)),
    "shadow": dict(shadow=(0.55, 0.75)),
    "far": dict(photo_w=(1500, 2100)),
    "cut": dict(cut=(0.12, 0.25)),
    "rotated": dict(rotation_deg=(8.0, 14.0), rotation_exact=True),
}
BAD = [c for c in CLASSES if c != "good"]


@dataclass
class Frame:
    name: str
    cls: str
    seed: int
    rgb: np.ndarray
    scale_to_capture: float
    params: dict


def _u(rng, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def make_frame(cls: str, seed: int) -> Frame:
    import cv2

    from ecgscan.bench.cases import bench_config
    from ecgscan.bench.render import GRID_PALETTES, RenderSpec, render_page
    from ecgscan.bench.sources import pick_preset, synthetic_signal
    from ecgscan.layout import get_layout

    rng = np.random.default_rng(seed)
    r = dict(GOOD, **CLASSES[cls])
    sig = synthetic_signal(pick_preset(bench_config()["sources"]["synthetic"]["adult"], rng), rng, seed)
    palette = ["red", "pink", "orange", "green", "gray"][int(rng.integers(0, 5))]
    assert palette in GRID_PALETTES
    page, _ = render_page(sig.leads_mv, sig.fs, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=PAGE_PPM, grid=palette))
    rot = _u(rng, r["rotation_deg"])
    spec = DegradeSpec(background=_u(rng, r["background"]), perspective=_u(rng, r["perspective"]),
                       rotation_deg=rot, noise=_u(rng, r["noise"]), vignette=_u(rng, r["vignette"]),
                       shadow=_u(rng, r["shadow"]), glare=_u(rng, r.get("glare", (0.0, 0.0))),
                       blur_px=_u(rng, r["blur_mm"]) * PAGE_PPM)
    if r.get("rotation_exact"):
        # degrade() draws the angle in [-max, max]: a fixed |angle| is applied here instead.
        spec = replace(spec, rotation_deg=0.0)
    img, _ = degrade(page, spec, seed=seed, px_per_mm=PAGE_PPM)
    if r.get("rotation_exact"):
        h, w = img.shape[:2]
        ang = rot * (1 if rng.random() < 0.5 else -1)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        c, s = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * s + w * c), int(h * c + w * s)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        img = cv2.warpAffine(img, m, (nw, nh), borderValue=(96, 112, 128))
    if "cut" in r:
        # The sheet runs out of the frame on one side.
        h, w = img.shape[:2]
        f = _u(rng, r["cut"])
        side = int(rng.integers(0, 4))
        img = [img[:, int(f * w):], img[:, : w - int(f * w)], img[int(f * h):], img[: h - int(f * h)]][side]
    img = np.clip(img.astype(np.float32) * _u(rng, r["exposure"]), 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    frame = cv2.resize(img, (FRAME_W, int(round(h * FRAME_W / w))), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    photo_w = _u(rng, r["photo_w"])
    params = {"palette": palette, **{k: round(v, 4) if isinstance(v, float) else v for k, v in spec.to_dict().items()},
              "rotation_deg": round(rot, 3), "photo_w": round(photo_w), "class_params": {k: list(v) if isinstance(v, tuple)
                                                                                         else v for k, v in CLASSES[cls].items()}}
    return Frame(f"{cls}-{seed:03d}", cls, seed, frame[..., ::-1].copy(), photo_w / FRAME_W, params)


def make_set(n_per_class: int, *, seed: int = 0, n_good: int | None = None) -> list[Frame]:
    out = []
    for i, cls in enumerate(CLASSES):
        n = n_good if (cls == "good" and n_good) else n_per_class
        out += [make_frame(cls, seed + 1000 * i + k) for k in range(n)]
    return out


def evaluate(frames: list[Frame], cfg: dict | None = None) -> dict:
    """Per class: how many frames the checks refuse (any hint) and which hints fired."""
    from ecgscan.capture_qc import check_frame

    rows = []
    for f in frames:
        q = check_frame(f.rgb, scale_to_capture=f.scale_to_capture, cfg=cfg)
        rows.append({"name": f.name, "cls": f.cls, "hints": q.hints, "metrics": q.to_dict(), "params": f.params})
    summary = {}
    for cls in CLASSES:
        rs = [r for r in rows if r["cls"] == cls]
        if not rs:
            continue
        refused = sum(1 for r in rs if r["hints"])
        hints: dict[str, int] = {}
        for r in rs:
            for h in r["hints"]:
                hints[h] = hints.get(h, 0) + 1
        summary[cls] = {"n": len(rs), "refused": refused, "share": round(refused / len(rs), 3), "hints": hints}
    bad = [r for r in rows if r["cls"] != "good"]
    good = [r for r in rows if r["cls"] == "good"]
    summary["_acceptance"] = {
        "bad_refused": round(sum(1 for r in bad if r["hints"]) / max(1, len(bad)), 3),
        "good_refused": round(sum(1 for r in good if r["hints"]) / max(1, len(good)), 3),
    }
    summary["_acceptance"]["passed"] = (summary["_acceptance"]["bad_refused"] >= 0.90
                                        and summary["_acceptance"]["good_refused"] <= 0.10)
    return {"summary": summary, "frames": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    import cv2

    ap = argparse.ArgumentParser(description="frame-check acceptance on synthetic good / bad frames (S1)")
    ap.add_argument("--n", type=int, default=20, help="frames per bad class")
    ap.add_argument("--good", type=int, default=60, help="good frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-frames", action="store_true")
    a = ap.parse_args(argv)
    frames = make_set(a.n, seed=a.seed, n_good=a.good)
    res = evaluate(frames)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "qc_frames.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if a.save_frames:
        for f in frames:
            cv2.imwrite(str(a.out / f"{f.name}.jpg"), f.rgb[..., ::-1])
    print(json.dumps(res["summary"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
