"""D182 photo check, step 2 (the model venv): ECG-FM's per-class probabilities on the source
record and on the record read from its photo, next to rhythm_hgb's (from photo_export.py).
The heads are the logistic regressions of probe.py, refitted on the cached training
embeddings. What is measured is how far each model's answer moves from record to photo.

    python tools/fm_probe/photo_probe.py --photo <dir from photo_export> --records <records.json> \
        --emb <probe_ecgfm.emb.npz> --model-dir <ecg-fm dir> --fairseq-signals <checkout> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    from probe import load_model
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--fairseq-signals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.photo / "meta.json").read_text())
    sig = np.load(a.photo / "signals.npz")
    data = json.loads(a.records.read_text())
    emb = dict(np.load(a.emb))
    tr = [r for r in data["records"] if r["split"] == "train" and r["name"] in emb]
    xtr = np.stack([emb[r["name"]] for r in tr])

    one = load_model("ecgfm", a.model_dir, a.fairseq_signals)
    e_orig = np.stack([one(x.astype(np.float64), 500.0) for x in sig["orig"]])
    e_photo = np.stack([one(x.astype(np.float64), 500.0) for x in sig["photo"]])

    gated = np.array([m["gated"] for m in meta])
    report = {"n": len(meta), "gated": int(gated.sum()), "classes": {}}
    sums = {k: [] for k in ("fm_shift", "hgb_shift", "fm_flip", "hgb_flip", "fm_rho", "hgb_rho")}
    for c in data["classes"]:
        ytr = np.array([c in r["gold"] for r in tr])
        hgb_o = np.array([m["hgb_orig"].get(c) for m in meta], dtype=object)
        if ytr.sum() < 10 or any(v is None for v in hgb_o):
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(xtr, ytr)
        fo, fp = clf.predict_proba(e_orig)[:, 1], clf.predict_proba(e_photo)[:, 1]
        ho = hgb_o.astype(float)
        hp = np.array([m["hgb_photo"][c] for m in meta], dtype=float)
        y = np.array([c in m["gold"] for m in meta])
        row = {"n_pos": int(y.sum()),
               "fm_shift": round(float(np.mean(np.abs(fp - fo))), 3),
               "hgb_shift": round(float(np.mean(np.abs(hp - ho))), 3),
               "fm_flip": round(float(np.mean((fp >= 0.5) != (fo >= 0.5))), 3),
               "hgb_flip": round(float(np.mean((hp >= 0.5) != (ho >= 0.5))), 3),
               "fm_rho": round(float(spearmanr(fo, fp).correlation), 3),
               "hgb_rho": round(float(spearmanr(ho, hp).correlation), 3)}
        if 5 <= y.sum() < len(y) - 5:
            row.update({k: round(float(roc_auc_score(y, v)), 3) for k, v in
                        (("fm_auroc_orig", fo), ("fm_auroc_photo", fp), ("hgb_auroc_orig", ho), ("hgb_auroc_photo", hp))})
        report["classes"][c] = row
        for k, vals in sums.items():
            if not np.isnan(row[k]):
                vals.append(row[k])
    report["mean"] = {k: round(float(np.mean(v)), 3) for k, v in sums.items()}
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean"]))


if __name__ == "__main__":
    main()
