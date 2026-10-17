"""Real photos with their digital originals (TZ §11 "золотая валидация", M5).

Folder layout (``data/real`` is never committed — the photos are of patients' ECGs)::

    data/real/<device>/device.yaml      # layout, paper speed, gain the device prints with
    data/real/<device>/<id>.jpg|png     # the photo (anonymised id; the header may be masked)
    data/real/<device>/<id>.xml|csv|edf|hea+dat   # the same recording, digital
    data/real/<device>/<id>.yaml        # optional per-record override (e.g. offset_s)

``device.yaml``::

    layout: 3x4+rhythm      # registry name
    speed_mm_s: 25
    gain_mm_mv: 10
    csv: {fs: 500, unit: uv}   # only for CSV originals (CSV carries no rate or unit)
    offset_s: 0.0           # where the printed 10 s start in the digital record

Digital formats: WFDB, EDF, CSV through the CardioLens loaders; GE MUSE XML here
(``<Waveform>`` of type Rhythm, base64 little-endian int16 per lead, scaled by
``LeadAmplitudeUnitsPerBit``). Limb leads the file lacks (MUSE stores I, II, V1-V6)
are derived from I and II. Philips XML is not read (compressed waveform; needs a
sample file).
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import yaml

from ecgscan.bench.cases import IMAGE_SUFFIXES, Case
from ecgscan.bench.render import PageGT, TraceGT, resample_to
from ecgscan.bench.sources import derive_limb_leads
from ecgscan.contract.models import STANDARD_LEADS
from ecgscan.layout import get_layout

GOLD_SUFFIXES = (".xml", ".csv", ".edf", ".hea")
_LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def read_muse_xml(path: str | Path) -> tuple[dict[str, np.ndarray], float]:
    """GE MUSE XML rhythm waveform -> ({lead: mV}, fs)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    waves = [w for w in root.iter("Waveform") if (w.findtext("WaveformType") or "").strip() == "Rhythm"]
    if not waves:
        raise ValueError(f"{path}: no Rhythm waveform")
    w = waves[0]
    fs = float(w.findtext("SampleBase"))
    out = {}
    for ld in w.iter("LeadData"):
        name = (ld.findtext("LeadID") or "").strip()
        name = _LEAD_ALIASES.get(name.upper(), name)
        raw = np.frombuffer(base64.b64decode(ld.findtext("WaveFormData") or ""), dtype="<i2").astype(np.float64)
        per_bit = float(ld.findtext("LeadAmplitudeUnitsPerBit") or 1.0)
        units = (ld.findtext("LeadAmplitudeUnits") or "MICROVOLTS").strip().upper()
        scale = {"MICROVOLTS": 1e-3, "MILLIVOLTS": 1.0}.get(units)
        if scale is None:
            raise ValueError(f"{path}: unknown amplitude units {units!r}")
        out[name] = raw * per_bit * scale
    return out, fs


def read_gold(path: str | Path, settings: dict) -> tuple[dict[str, np.ndarray], float]:
    """Any supported digital original -> ({lead: mV}, fs)."""
    path = Path(path)
    if path.suffix.lower() == ".xml":
        leads, fs = read_muse_xml(path)
    else:
        from cardiolens.layer1_ingestion import load

        kw = {}
        if path.suffix.lower() == ".csv":
            c = settings.get("csv") or {}
            if "fs" not in c:
                raise ValueError(f"{path}: a CSV original needs csv.fs in device.yaml (CSV carries no rate)")
            kw = {"sampling_rate_hz": float(c["fs"]), "unit": c.get("unit", "uv")}
        sig = load(path, **kw)
        fs = float(sig.sampling_rate_hz)
        leads = {n: sig.signal[i].astype(np.float64) / 1000.0 for i, n in enumerate(sig.lead_names)}
    if "I" in leads and "II" in leads and not {"III", "aVR", "aVL", "aVF"} <= set(leads):
        derived = derive_limb_leads(leads)
        leads = {**derived, **leads}
    return {k: v for k, v in leads.items() if k in STANDARD_LEADS}, fs


def _settings(folder: Path, stem: str) -> dict:
    s = {}
    for p in (folder / "device.yaml", folder / f"{stem}.yaml"):
        if p.exists():
            s.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return s


def real_cases(root: str | Path, *, fs: float = 500.0) -> list[Case]:
    """Every photo under ``root/<device>/`` with a digital original of the same id."""
    out = []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            gold = next((img.with_suffix(s) for s in GOLD_SUFFIXES if img.with_suffix(s).exists()), None)
            if gold is None:
                continue
            st = _settings(folder, img.stem)
            lay = get_layout(st.get("layout", "3x4+rhythm"))
            leads, gold_fs = read_gold(gold, st)
            sig = {k: resample_to(v, gold_fs, fs) for k, v in leads.items()}
            i0 = int(round(float(st.get("offset_s", 0.0)) * fs))
            traces = []
            for lead, (t0, t1) in lay.lead_windows().items():
                if lead in sig:
                    traces.append(TraceGT(lead, "panel", t0, sig[lead][i0 + int(t0 * fs): i0 + int(t1 * fs)], 0.0, 0.0))
            for lead in lay.rhythm:
                if lead in sig:
                    traces.append(TraceGT(lead, "rhythm", 0.0, sig[lead][i0: i0 + int(lay.record_s * fs)], 0.0, 0.0))
            truth = PageGT(layout=lay.name, speed_mm_s=float(st.get("speed_mm_s", 25)),
                           gain_mm_mv=float(st.get("gain_mm_mv", 10)), px_per_mm=0.0, fs=fs, width_px=0,
                           height_px=0, header_box_px=None, traces=traces)
            case_id = f"{folder.name}/{img.stem}"
            meta = {"case_id": case_id, "slice": f"real_{folder.name}", "layout": lay.name,
                    "source": {"id": gold.name, "format": gold.suffix.lower().lstrip(".")},
                    "assumed": {k: st.get(k) for k in ("layout", "speed_mm_s", "gain_mm_mv", "offset_s")},
                    "true_px_per_mm": None}
            out.append(Case(case_id, f"real_{folder.name}", img, truth, meta))
    return out
