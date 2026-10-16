"""Bench cases: generate a suite from ``configs/scanner/bench.yaml`` and load it back.

A case directory holds ``image.png|jpg`` (what a digitiser sees), ``truth.npz``
(the true signal behind every drawn trace, mV at ``fs``) and ``case.json``
(slice, source, render and degrade settings, page geometry, page -> photo map).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecgscan.bench.degrade import DegradeSpec, degrade
from ecgscan.bench.render import PageGT, RenderSpec, TraceGT, render_page
from ecgscan.bench.sources import (
    SourceSignal,
    pick_preset,
    ptbxl_records,
    ptbxl_signal,
    synthetic_signal,
)
from ecgscan.config import load_yaml, repo_root
from ecgscan.layout import get_layout


@dataclass
class Case:
    case_id: str
    slice: str
    image_path: Path
    truth: PageGT
    meta: dict


def bench_config() -> dict:
    return load_yaml("bench.yaml")


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def slice_settings(cfg: dict, name: str, suite: dict | None = None) -> dict:
    s = _merge(cfg["defaults"], cfg["slices"][name])
    if suite and suite.get("render"):
        s["render"] = _merge(s["render"], suite["render"])
    return s


def suite_slices(cfg: dict, suite_name: str) -> tuple[dict, list[str]]:
    suite = cfg["suites"][suite_name]
    names = list(cfg["slices"]) if suite.get("slices") == "all" else list(suite["slices"])
    return suite, names


def case_seed(base: int, slice_name: str, i: int) -> int:
    h = hashlib.sha256(f"{base}:{slice_name}:{i}".encode()).hexdigest()
    return int(h[:8], 16)


def _source(cfg: dict, kind: str, rng: np.random.Generator, seed: int, i: int) -> SourceSignal | None:
    if kind == "ptbxl":
        recs = ptbxl_records(repo_root() / cfg["sources"]["ptbxl_dir"])
        return ptbxl_signal(recs[(seed + i) % len(recs)]) if recs else None
    presets = cfg["sources"]["synthetic"][kind]
    return synthetic_signal(pick_preset(presets, rng), rng, seed % 100000)


def generate_suite(suite_name: str, out_dir: str | Path, *, cases_per_slice: int | None = None,
                   slices: list[str] | None = None) -> tuple[list[Case], dict[str, str]]:
    """Render every case of a suite; returns (cases, {slice: reason}) for skipped slices."""
    import cv2

    cfg = bench_config()
    suite, names = suite_slices(cfg, suite_name)
    names = slices or names
    n_per = cases_per_slice or int(suite["cases_per_slice"])
    out_dir = Path(out_dir)
    cases, skipped = [], {}
    for name in names:
        st = slice_settings(cfg, name, suite)
        layout = get_layout(st["layout"])
        rspec, dspec = RenderSpec(**st["render"]), DegradeSpec(**st["degrade"])
        for i in range(n_per):
            seed = case_seed(int(cfg["seed"]), name, i)
            rng = np.random.default_rng(seed)
            src = _source(cfg, st["source"], rng, seed, i)
            if src is None:
                skipped[name] = f"source '{st['source']}' not available ({cfg['sources']['ptbxl_dir']} is empty)"
                break
            page, truth = render_page(src.leads_mv, src.fs, layout, rspec, fs=float(cfg["fs_hz"]))
            img, info = degrade(page, dspec, seed=seed, px_per_mm=rspec.px_per_mm)
            cid = f"{name}-{i:03d}"
            cdir = out_dir / name / cid
            cdir.mkdir(parents=True, exist_ok=True)
            img_path = cdir / ("image.jpg" if dspec.jpeg_quality is not None else "image.png")
            if img_path.suffix == ".jpg":
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # already JPEG-degraded
            else:
                cv2.imwrite(str(img_path), img)
            np.savez_compressed(cdir / "truth.npz", **{f"t{k}": t.samples_mv.astype(np.float32)
                                                        for k, t in enumerate(truth.traces)})
            meta = {"case_id": cid, "slice": name, "seed": seed, "layout": layout.name,
                    "source": {"id": src.source_id, **src.meta}, "render": rspec.to_dict(),
                    "degrade": dspec.to_dict(), "page": truth.meta(), "photo": info,
                    "true_px_per_mm": round(rspec.px_per_mm * (info["centre_scale"] or 1.0), 3)}
            (cdir / "case.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            cases.append(Case(cid, name, img_path, truth, meta))
    return cases, skipped


def load_case(cdir: str | Path) -> Case:
    cdir = Path(cdir)
    meta = json.loads((cdir / "case.json").read_text(encoding="utf-8"))
    arrays = np.load(cdir / "truth.npz")
    p = meta["page"]
    traces = [TraceGT(lead=t["lead"], kind=t["kind"], start_s=t["start_s"],
                      samples_mv=arrays[f"t{k}"].astype(np.float64), x0_px=t["x0_px"], baseline_px=t["baseline_px"])
              for k, t in enumerate(p["traces"])]
    truth = PageGT(layout=p["layout"], speed_mm_s=p["speed_mm_s"], gain_mm_mv=p["gain_mm_mv"],
                   px_per_mm=p["px_per_mm"], fs=p["fs"], width_px=p["width_px"], height_px=p["height_px"],
                   header_box_px=tuple(p["header_box_px"]) if p["header_box_px"] else None, traces=traces)
    img = next(cdir.glob("image.*"))
    return Case(meta["case_id"], meta["slice"], img, truth, meta)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def image_record_cases(folder: str | Path, *, layout: str = "3x4+rhythm", speed_mm_s: float = 25.0,
                       gain_mm_mv: float = 10.0, fs: float = 500.0, slice_name: str = "image_folder") -> list[Case]:
    """Real or externally generated images paired with their WFDB records.

    Layouts: ECG-Image-Database / PhysioNet Challenge 2024 (``<rec>-0001.png``) and
    ECG-Image-Kit output (``<rec>-0.png``), next to ``<rec>.hea/.dat``. The page layout
    and calibration are not recorded per image there; they are the stated
    assumptions (``layout``, ``speed_mm_s``, ``gain_mm_mv``), written into each
    case's meta so the report can say what the truth rests on.
    """
    import re

    from ecgscan.bench.render import resample_to
    from ecgscan.bench.sources import ptbxl_signal

    folder = Path(folder)
    records = {p.stem: p.with_suffix("") for p in folder.rglob("*.hea") if p.with_suffix(".dat").exists()}
    lay = get_layout(layout)
    out = []
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        rec = records.get(img.stem) or records.get(re.sub(r"-\d+$", "", img.stem))
        if rec is None:
            continue
        src = ptbxl_signal(rec)
        sig = {k: resample_to(v, src.fs, fs) for k, v in src.leads_mv.items()}
        traces = []
        for lead, (t0, t1) in lay.lead_windows().items():
            if lead in sig:
                traces.append(TraceGT(lead, "panel", t0, sig[lead][int(t0 * fs):int(t1 * fs)], 0.0, 0.0))
        for lead in lay.rhythm:
            if lead in sig:
                traces.append(TraceGT(lead, "rhythm", 0.0, sig[lead][:int(lay.record_s * fs)], 0.0, 0.0))
        truth = PageGT(layout=lay.name, speed_mm_s=speed_mm_s, gain_mm_mv=gain_mm_mv, px_per_mm=0.0, fs=fs,
                       width_px=0, height_px=0, header_box_px=None, traces=traces)
        meta = {"case_id": img.stem, "slice": slice_name, "layout": lay.name, "source": {"id": str(rec)},
                "assumed": {"layout": lay.name, "speed_mm_s": speed_mm_s, "gain_mm_mv": gain_mm_mv},
                "true_px_per_mm": None}
        out.append(Case(img.stem, slice_name, img, truth, meta))
    return out


def load_suite(root: str | Path) -> list[Case]:
    return [load_case(p.parent) for p in sorted(Path(root).rglob("case.json"))]
