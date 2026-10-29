"""D182 photo check, step 1 (project venv): for each ECG-Image-Kit bench photo (D181), the
source record and the record read from the photo through the site's route, as 12 x 5000
arrays at 500 Hz, with rhythm_hgb's probabilities on each.

    python tools/fm_probe/photo_export.py --bench <dir> [<dir> ...] --records <records.json> --out <dir>

A lead the photo route did not read is zeros; a digitised 3x4 page keeps its bridged samples
outside each lead's window — what a model on the site would be given.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sps

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible", "photo_coarse"}


def to_grid(sig) -> np.ndarray:
    """A SignalRepresentation as 12 x 5000 (500 Hz, 10 s), leads in the standard order."""
    x = np.asarray(sig.signal, dtype=np.float64)
    fs = float(sig.sampling_rate_hz)
    if fs != 500.0:
        x = sps.resample(x, int(round(x.shape[1] * 500.0 / fs)), axis=1)
    out = np.zeros((12, 5000), dtype=np.float32)
    for i, lead in enumerate(LEADS):
        if lead in sig.lead_names:
            row = np.nan_to_num(x[sig.lead_names.index(lead)][:5000])
            out[i, :len(row)] = row
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.ml.features import features_from_result
    from cardiolens.ml.model import load_model
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--model", default="models/cardiolens/rhythm_hgb.joblib")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    model = load_model(a.model)
    gold = {r["name"]: r for r in json.loads(a.records.read_text())["records"]}
    a.out.mkdir(parents=True, exist_ok=True)
    meta, orig, photo = [], [], []
    for bench in a.bench:
        for png in sorted((bench / "images").glob("*-0.png")):
            rec = png.name.rsplit("-", 1)[0]
            name = f"ptbxl_{int(rec.split('_')[0])}"
            if name not in gold:
                continue
            try:
                ro = run_pipeline(load_wfdb(str(bench / "records" / rec)), mode="A")
                rp = run_pipeline(digitize_photos([png]), mode="A")
            except Exception as e:  # a photo the route refuses is left out, and said
                print(f"skip {bench.name}/{rec}: {str(e)[:80]}", file=sys.stderr)
                continue
            meta.append({"name": name, "set": bench.name, "split": gold[name]["split"], "gold": gold[name]["gold"],
                         "gated": bool(rp.findings.codes() & GATES),
                         "hgb_orig": model.predict_proba(features_from_result(ro)),
                         "hgb_photo": model.predict_proba(features_from_result(rp))})
            orig.append(to_grid(ro.signal))
            photo.append(to_grid(rp.signal))
            print(f"{len(meta)} {bench.name}/{rec} gated={meta[-1]['gated']}", file=sys.stderr, flush=True)
    np.savez_compressed(a.out / "signals.npz", orig=np.stack(orig), photo=np.stack(photo))
    (a.out / "meta.json").write_text(json.dumps(meta))
    print(f"{len(meta)} photos; gated {sum(m['gated'] for m in meta)}", file=sys.stderr)


if __name__ == "__main__":
    main()
