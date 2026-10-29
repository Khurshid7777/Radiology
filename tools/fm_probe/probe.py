"""D182 step 2 (a separate venv with torch CPU, transformers, scikit-learn, wfdb, scipy):
frozen HuBERT-ECG or ECG-FM embeddings -> one logistic regression per class -> AUROC on
the hold-outs, next to rhythm_hgb's AUROC on the same records.

ECG-FM (``--backend ecgfm``, MIT) loads through fairseq-signals (``--fairseq-signals``,
a checkout of github.com/Jwoo5/fairseq-signals; transformers < 4.50) with the authors'
preprocessing (bowang-lab/ECG-FM notebooks/infer_quickstart): 500 Hz, standard lead
order, z-score per lead, 5-s non-overlapping windows; the encoder output is averaged.

    python tools/fm_probe/probe.py --records <dir>/records.json --model-dir <hubert-ecg-small dir> \
        --out <dir>/probe.json

Preprocessing as the authors' hubert_ecg/utils.py: FIR band-pass 0.05-47 Hz, 500 Hz,
min-max to [-1, 1] per lead, 5-s windows, the 12 leads flattened and decimated by 5
(6000 samples); a 10-s record is two windows, their embeddings averaged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

FS = 500
WIN = 5 * FS


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    """(12, n) at ``fs`` -> (12, m) at 500 Hz, band-passed and scaled to [-1, 1]."""
    order = int(0.3 * fs)
    order += order % 2 == 0          # a band-pass FIR needs an odd number of taps
    taps = signal.firwin(order, [0.05, 47.0], pass_zero=False, fs=fs)
    x = signal.filtfilt(taps, [1.0], x, axis=1)
    if fs != FS:
        x = signal.resample(x, int(x.shape[1] * FS / fs), axis=1)
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return 2 * (x - lo) / (hi - lo + 1e-8) - 1


def windows(x: np.ndarray) -> list[np.ndarray]:
    """5-s windows of a (12, n) record, each flattened and decimated to 6000 samples."""
    out = []
    for s in range(0, max(1, x.shape[1] - WIN + 1), WIN):
        w = x[:, s:s + WIN]
        if w.shape[1] < WIN:
            w = np.pad(w, ((0, 0), (0, WIN - w.shape[1])))
        out.append(signal.decimate(w.reshape(-1), 5).astype(np.float32))
    return out


def load(path: str) -> tuple[np.ndarray, float]:
    import wfdb
    rec = wfdb.rdrecord(path)
    x = np.nan_to_num(rec.p_signal.T.astype(np.float64))
    if x.shape[0] != 12:
        raise ValueError(f"{path}: {x.shape[0]} leads")
    return x, float(rec.fs)


def ecgfm_windows(x: np.ndarray, fs: float) -> np.ndarray:
    """(12, n) -> (W, 12, 2500): 500 Hz, z-score per lead (a flat lead stays 0), 5-s windows."""
    if fs != FS:
        x = signal.resample(x, int(x.shape[1] * FS / fs), axis=1)
    sd = x.std(axis=1, keepdims=True)
    x = np.where(sd > 1e-8, (x - x.mean(axis=1, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0), 0.0)
    n = x.shape[1] // WIN
    return np.stack([x[:, i * WIN:(i + 1) * WIN] for i in range(max(n, 1))]).astype(np.float32)


def load_model(backend: str, model_dir: Path, fairseq_signals: Path | None):
    """The model and a function (record, fs) -> one embedding vector."""
    import torch
    if backend == "ecgfm":
        sys.path.insert(0, str(fairseq_signals))
        from fairseq_signals.models import build_model_from_checkpoint
        ckpt = next(model_dir.glob("*finetuned.pt"), None) or next(model_dir.glob("*.pt"))
        model = build_model_from_checkpoint(checkpoint_path=str(ckpt)).eval()

        def one(x, fs):
            with torch.no_grad():
                h = model(source=torch.from_numpy(ecgfm_windows(x, fs)), padding_mask=None)["encoder_out"]
            return h.mean(dim=(0, 1)).numpy()
        return one
    sys.path.insert(0, str(model_dir))
    from hubert_ecg import HuBERTECG  # the model's own file, read before use (D182)

    model = HuBERTECG.from_pretrained(str(model_dir)).eval()

    def one(x, fs):
        ws = np.stack(windows(preprocess(x, fs)))
        with torch.no_grad():
            h = model(torch.from_numpy(ws)).last_hidden_state      # (W, T, D)
        return h.mean(dim=(0, 1)).numpy()
    return one


def embed(records: list[dict], one, cache: Path) -> np.ndarray:
    done = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    for i, r in enumerate(records):
        if r["name"] in done:
            continue
        try:
            x, fs = load(r["path"])
            done[r["name"]] = one(x, fs)
        except Exception as e:    # an unreadable record is left out, and said
            print(f"skip {r['name']}: {e}", file=sys.stderr)
            done[r["name"]] = None
        if (i + 1) % 200 == 0:
            np.savez(cache, **{k: v for k, v in done.items() if v is not None})
            print(f"{i + 1}/{len(records)}", file=sys.stderr, flush=True)
    np.savez(cache, **{k: v for k, v in done.items() if v is not None})
    return {k: v for k, v in done.items() if v is not None}


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", choices=("hubert", "ecgfm"), default="hubert")
    ap.add_argument("--fairseq-signals", type=Path, default=None)
    ap.add_argument("--min-pos", type=int, default=5, help="hold-out positives a class needs to be scored")
    a = ap.parse_args()
    data = json.loads(a.records.read_text())
    recs, classes = data["records"], data["classes"]
    emb = embed(recs, load_model(a.backend, a.model_dir, a.fairseq_signals), a.out.with_suffix(".emb.npz"))
    recs = [r for r in recs if r["name"] in emb]
    tr = [r for r in recs if r["split"] == "train"]
    xtr = np.stack([emb[r["name"]] for r in tr])
    report = {"model": a.model_dir.name, "backend": a.backend, "n_train": len(tr), "classes": {}}
    for split in ("holdout_ecgarr", "holdout_ptbxl"):
        te = [r for r in recs if r["split"] == split]
        xte = np.stack([emb[r["name"]] for r in te])
        rows, fm_aucs, hgb_aucs = {}, [], []
        for c in classes:
            ytr = np.array([c in r["gold"] for r in tr])
            yte = np.array([c in r["gold"] for r in te])
            if ytr.sum() < 10 or yte.sum() < a.min_pos or yte.all():
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
            clf.fit(xtr, ytr)
            fm = roc_auc_score(yte, clf.predict_proba(xte)[:, 1])
            hgb_p = [r["hgb"].get(c) for r in te]
            hgb = roc_auc_score(yte, hgb_p) if None not in hgb_p else None
            rows[c] = {"n_pos": int(yte.sum()), "fm_auroc": round(fm, 3),
                       "hgb_auroc": round(hgb, 3) if hgb is not None else None}
            if hgb is not None:
                fm_aucs.append(fm)
                hgb_aucs.append(hgb)
        report[split] = {"n": len(te), "macro_fm": round(float(np.mean(fm_aucs)), 3),
                         "macro_hgb": round(float(np.mean(hgb_aucs)), 3), "per_class": rows}
        print(split, len(te), "macro FM", report[split]["macro_fm"], "macro hgb", report[split]["macro_hgb"])
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
