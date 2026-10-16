"""Signal sources for bench cases: synthetic presets and local PTB-XL records (TZ §11).

The bench depends on CardioLens for its synthetic generator and WFDB reader (a
dev-time dependency; the ``ecgscan`` core does not import ``cardiolens``).
Signals are returned as ``{lead: mV}`` at their own sampling rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SourceSignal:
    source_id: str
    fs: float
    leads_mv: dict[str, np.ndarray]
    meta: dict


def synthetic_signal(preset: dict, rng: np.random.Generator, case_seed: int) -> SourceSignal:
    """One record from a preset (``configs/scanner/bench.yaml: sources.synthetic``)."""
    from cardiolens.layer1_ingestion import generate_synthetic_ecg

    kw = {k: v for k, v in preset.items() if k not in ("name", "heart_rate_bpm", "weight")}
    hr = preset.get("heart_rate_bpm", 72)
    hr = float(rng.uniform(*hr)) if isinstance(hr, list | tuple) else float(hr)
    sig = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10.0, sampling_rate_hz=500.0,
                                 seed=case_seed, **kw)
    leads = {n: sig.lead(n).astype(np.float64) / 1000.0 for n in sig.lead_names}
    return SourceSignal(source_id=f"synthetic:{preset['name']}:{case_seed}", fs=sig.sampling_rate_hz,
                        leads_mv=leads, meta={"preset": preset["name"], "heart_rate_bpm": round(hr, 1),
                                              "age_years": kw.get("age_years", 40.0)})


def ptbxl_records(root: str | Path) -> list[Path]:
    """WFDB records (path without extension) under a local PTB-XL folder (``fetch_ptbxl`` output)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.with_suffix("") for p in root.rglob("*.hea") if p.with_suffix(".dat").exists())


def ptbxl_signal(record: Path) -> SourceSignal:
    from cardiolens.layer1_ingestion import load_wfdb

    sig = load_wfdb(record)
    leads = {n: sig.lead(n).astype(np.float64) / 1000.0 for n in sig.lead_names}
    return SourceSignal(source_id=f"ptbxl:{record.name}", fs=sig.sampling_rate_hz, leads_mv=leads,
                        meta={"record": record.name, "age_years": sig.patient.age_years})


PTBXL_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
PTBXL_MAX_ID = 21837


def fetch_ptbxl_sample(out: str | Path, n: int = 24) -> list[Path]:
    """Download ``n`` evenly spaced 500-Hz PTB-XL records (no label balancing — for
    image benchmarking any record will do). Licensed data: ``data/ptbxl`` is git-ignored."""
    from cardiolens.validation.fetch_physionet import fetch_bytes

    out = Path(out)
    got = []
    for rid in np.linspace(1, PTBXL_MAX_ID, n).astype(int):
        rel = f"records500/{rid // 1000 * 1000:05d}/{rid:05d}_hr"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            for ext in (".hea", ".dat"):
                if not dst.with_suffix(ext).exists():
                    dst.with_suffix(ext).write_bytes(fetch_bytes(PTBXL_URL + rel + ext))
        except Exception:  # a missing id (PTB-XL has gaps) is skipped
            for ext in (".hea", ".dat"):
                dst.with_suffix(ext).unlink(missing_ok=True)
            continue
        got.append(dst)
    return got


def pick_preset(presets: list[dict], rng: np.random.Generator) -> dict:
    w = np.array([float(p.get("weight", 1.0)) for p in presets])
    return presets[int(rng.choice(len(presets), p=w / w.sum()))]
