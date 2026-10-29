"""S3 masking bench (TZ 18.6, 18.14: personal data masked in 100 % of test cases).

A synthetic sheet is a printed header (``bench.headers``: its name / ID / birth-date lines and
their boxes known) over an ECG page; it is photographed (perspective, tilt, quarter turns) and
the doctor's corners are placed with a small error. Masked as the app does it:

* ``profile_zones``: the recorder's ``phi_zones`` mapped through the corners (``ecgscan.masking``,
  the app's twin) -- the mode a profile gives;
* ``band``: the top band of the photo (the app's default share) -- the mode without a profile,
  which the doctor adjusts on the review screen.

A case passes when every personal-data line is covered completely (>= ``need`` of its pixels).
A contact sheet of masked photos is saved for the visual check the TZ asks for.

    python -m ecgscan.bench.mask_bench --n 60 --out reports/scanner/s3-masking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecgscan.bench.headers import render_header
from ecgscan.masking import mask_zones, sheet_corners, zone_polygon

# The synthetic recorder's profile: its header prints name / ID / birth date in the top-left.
ZONES = [[0.0, 0.0, 0.72, 0.105]]
MARGIN = 0.01
APP_BAND = 0.14                      # mobile Settings.headerBand default


def _sheet(seed: int, ppm: float):
    """Header over an ECG page, as one upright sheet; the personal-data boxes in sheet pixels."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.bench.sources import derive_limb_leads
    from ecgscan.layout import get_layout

    ecg = generate_synthetic_ecg(heart_rate_bpm=70 + seed % 30, seed=seed)
    mv = derive_limb_leads({n: ecg.lead(n).astype(np.float64) / 1000.0 for n in ecg.lead_names})
    page, _ = render_page(mv, 500.0, get_layout("3x4+rhythm"), RenderSpec(px_per_mm=ppm, header=False))
    hdr = render_header(seed, px_per_mm=ppm, degrade={})
    w = page.shape[1]
    top = np.full((hdr.image.shape[0], w, 3), 255, np.uint8)
    top[:, :min(w, hdr.image.shape[1])] = hdr.image[:, :w]
    sheet = np.vstack([top, page])
    phi = [ln.box for ln in hdr.lines if ln.phi]
    return sheet, phi, hdr


def _photo(sheet: np.ndarray, rng: np.random.Generator, turn: int):
    """The sheet photographed: perspective and tilt, turned ``turn`` quarter turns counter-clockwise
    in the photo (the doctor turns it back clockwise: ``rotate_deg = 90 * turn``)."""
    import cv2

    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pad = 0.08 * max(w, h)
    jit = rng.uniform(-0.04, 0.04, (4, 2)) * [w, h]
    dst = src + pad + jit
    m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    size = (int(w + 2 * pad), int(h + 2 * pad))
    photo = cv2.warpPerspective(sheet, m, size, borderValue=(70, 80, 90))
    for _ in range(turn):
        photo = np.ascontiguousarray(np.rot90(photo))          # counter-clockwise
        m = np.array([[0, 1, 0], [-1, 0, size[0]], [0, 0, 1]], float) @ m
        size = (size[1], size[0])
    return photo, m


def _apply(m: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ m.T
    return p[:, :2] / p[:, 2:]


def run(n: int, out: Path, seed: int = 0, ppm: float = 10.0, need: float = 0.999) -> dict:
    import cv2

    rng = np.random.default_rng(seed)
    rows, thumbs, heads = [], [], []
    for i in range(n):
        s = int(rng.integers(2**31))
        sheet, phi, hdr = _sheet(s, ppm)
        turn = int(rng.choice([0, 0, 0, 1, 3]))
        photo, m = _photo(sheet, rng, turn)
        h, w = sheet.shape[:2]
        sheet_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        true_corners = _apply(m, sheet_px)                       # TL, TR, BR, BL of the sheet in the photo
        # The doctor's corners: the photo's quad listed from the photo's top-left, ~0.5 % off.
        k = turn % 4
        quad = [true_corners[(i2 + k) % 4] for i2 in range(4)]   # photo order: undo the turn
        quad = [list(map(float, p + rng.normal(0, 0.005 * max(w, h), 2))) for p in quad]
        rotate = (90 * turn) % 360
        assert np.allclose(sheet_corners(quad, rotate), [list(true_corners[j]) for j in range(4)], atol=0.03 * max(w, h))

        zoned = mask_zones(photo, ZONES, quad, rotate, MARGIN)
        zmask = np.zeros(photo.shape[:2], np.uint8)
        for z in ZONES:
            cv2.fillPoly(zmask, [np.round(np.array(zone_polygon(z, quad, rotate, MARGIN))).astype(np.int32)], 1)
        bmask = np.zeros(photo.shape[:2], np.uint8)
        bmask[:int(round(photo.shape[0] * APP_BAND))] = 1

        cov_z, cov_b = [], []
        for x0, y0, x1, y1 in phi:
            box = np.zeros(photo.shape[:2], np.uint8)
            cv2.fillPoly(box, [np.round(_apply(m, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])).astype(np.int32)], 1)
            area = max(1, int(box.sum()))
            cov_z.append(float((box & zmask).sum()) / area)
            cov_b.append(float((box & bmask).sum()) / area)
        # The trace area the zones take away (they must not reach the first row).
        first_row = _apply(m, [[0, hdr.image.shape[0] + 0.0], [w, hdr.image.shape[0] + 0.0]])
        phi_bottom = max(float(_apply(m, [[x0, y1], [x1, y1]])[:, 1].max()) for x0, y0, x1, y1 in phi)
        rows.append(dict(case=hdr.case_id, turn_deg=rotate, phi_lines=len(phi), zones_min_cover=round(min(cov_z), 4),
                         band_needed=round(phi_bottom / photo.shape[0], 3) if rotate == 0 else None,
                         band_min_cover=round(min(cov_b), 4), zones_ok=min(cov_z) >= need, band_ok=min(cov_b) >= need,
                         zones_reach_traces=bool(zmask[np.clip(first_row[:, 1].astype(int) + 5, 0, zmask.shape[0] - 1),
                                                       np.clip(first_row[:, 0].astype(int), 0, zmask.shape[1] - 1)].any())))
        # For the visual check: the masked photo mapped back onto the upright sheet, its header.
        back = cv2.warpPerspective(zoned, np.linalg.inv(m), (w, h), borderValue=(70, 80, 90))
        heads.append(cv2.resize(back[:hdr.image.shape[0] + 40], (1300, int(1300 * (hdr.image.shape[0] + 40) / w))))
        if len(thumbs) < 12:
            t = cv2.resize(zoned, (480, int(480 * zoned.shape[0] / zoned.shape[1])))
            thumbs.append(cv2.copyMakeBorder(t, 0, max(0, 700 - t.shape[0]), 4, 4, cv2.BORDER_CONSTANT,
                                             value=(255, 255, 255))[:700])
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(n=n, seed=seed, zones=ZONES, margin=MARGIN, band=APP_BAND, need=need,
                   zones_ok=sum(r["zones_ok"] for r in rows), band_ok=sum(r["band_ok"] for r in rows),
                   band_ok_upright=sum(r["band_ok"] for r in rows if r["turn_deg"] == 0),
                   upright=sum(r["turn_deg"] == 0 for r in rows),
                   zones_reach_traces=sum(r["zones_reach_traces"] for r in rows),
                   zones_min_cover=min(r["zones_min_cover"] for r in rows),
                   band_needed_upright=sorted(r["band_needed"] for r in rows if r["band_needed"] is not None))
    (out / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if thumbs:
        grid = np.vstack([np.hstack(thumbs[j:j + 4]) for j in range(0, len(thumbs) - len(thumbs) % 4, 4)])
        cv2.imwrite(str(out / "contact_sheet_zones.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    for j in range(0, len(heads), 10):
        cv2.imwrite(str(out / f"headers_masked_{j // 10:02d}.jpg"), np.vstack(
            [cv2.copyMakeBorder(x, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for x in heads[j:j + 10]]),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="personal-data masking on synthetic sheet photos (S3)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.n, a.out, a.seed), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
