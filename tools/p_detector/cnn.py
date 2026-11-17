"""D295: P-wave detector — a small 1-D CNN trained on manual P annotations (LUDB, QT Database).

    python tools/p_detector/cnn.py prep  --out <dir>          # build training / test arrays (numbers only)
    python tools/p_detector/cnn.py train --data <dir> --out models/p_detector/p_cnn.onnx
    python tools/p_detector/cnn.py eval  --data <dir> --model models/p_detector/p_cnn.onnx

Split (D295, fixed before training): train = even LUDB ids (12 leads) + even-index QTDB records (manually annotated
beats only); test = odd LUDB ids, odd QTDB records. Input: one lead at 500 Hz, band 0.5-40 Hz, scaled by its robust SD.
Output: per-sample probability of "inside a P wave" (onset..offset).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FS = 500
WIN = 2048
QTDB_SHARE = 0.4      # D295 it.2: QTDB crops share (it.1 sampled items uniformly: ~8 % QTDB)
LUDB_LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")


def prep_signal(x: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt, resample_poly
    x = np.asarray(x, float)
    if fs != FS:
        x = resample_poly(x, FS, int(fs))
    b, a = butter(2, [0.5 / (FS / 2), 40.0 / (FS / 2)], btype="band")
    y = filtfilt(b, a, x)
    s = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-6
    return (y / s).astype(np.float32)


def p_spans(ann, scale: float = 1.0):
    """(onset, peak, offset) of each P from '(' 'p' ')' triplets."""
    sym, smp = ann.symbol, ann.sample
    out = []
    for i, s in enumerate(sym):
        if s == "p" and 0 < i < len(sym) - 1 and sym[i - 1] == "(" and sym[i + 1] == ")":
            out.append((int(smp[i - 1] * scale), int(smp[i] * scale), int(smp[i + 1] * scale)))
    return out


def beat_spans(ann, scale: float = 1.0):
    """Annotated region per QTDB beat: from its P onset (or QRS onset - 300 ms) to QRS onset + 120 ms."""
    sym, smp = ann.symbol, ann.sample
    out = []
    for i, s in enumerate(sym):
        if s == "N" and i > 0 and sym[i - 1] == "(":
            on = int(smp[i - 1] * scale)
            out.append((on - int(0.35 * FS), on + int(0.12 * FS)))
    return out


def prep(out: Path) -> None:
    import wfdb
    out.mkdir(parents=True, exist_ok=True)
    items = []        # (split, source, id, lead, signal, label, mask, gold_peaks)
    for hea in sorted((ROOT / "data/ludb/data").glob("*.hea")):
        rid = int(hea.stem)
        rec = wfdb.rdrecord(str(hea)[:-4])
        for li, ln in enumerate(rec.sig_name):
            ln = ln.lower()
            if ln not in LUDB_LEADS:
                continue
            try:
                ann = wfdb.rdann(str(hea)[:-4], ln)
            except Exception:
                continue
            x = prep_signal(rec.p_signal[:, li], rec.fs)
            lab = np.zeros(x.size, np.uint8)
            mask = np.zeros(x.size, np.uint8)
            mask[max(0, int(ann.sample.min()) - 50):min(x.size, int(ann.sample.max()) + 50)] = 1
            peaks = []
            for on, pk, off in p_spans(ann):
                lab[on:off + 1] = 1
                peaks.append(pk)
            items.append(dict(split=("val" if rid % 4 == 0 else "train") if rid % 2 == 0 else "test", src="ludb", id=rid, lead=ln,
                              x=x, y=lab, m=mask, peaks=np.array(peaks, int),
                              lo=int(ann.sample.min()), hi=int(ann.sample.max())))
    recs = (ROOT / "data/qtdb/RECORDS").read_text().split()
    for k, r in enumerate(recs):
        p = ROOT / "data/qtdb" / r
        if not (p.with_suffix(".q1c")).exists():
            continue
        try:
            rec = wfdb.rdrecord(str(p))
            ann = wfdb.rdann(str(p), "q1c")
        except Exception:
            continue
        sc = FS / rec.fs
        spans = beat_spans(ann, sc)
        ps = p_spans(ann, sc)
        for li in range(rec.n_sig):
            x = prep_signal(rec.p_signal[:, li], rec.fs)
            lab = np.zeros(x.size, np.uint8)
            mask = np.zeros(x.size, np.uint8)
            for a, b in spans:
                mask[max(0, a):min(x.size, b)] = 1
            for on, pk, off in ps:
                lab[on:off + 1] = 1
            items.append(dict(split=("val" if k % 4 == 0 else "train") if k % 2 == 0 else "test", src="qtdb", id=r, lead=f"ch{li}",
                              x=x, y=lab, m=mask, peaks=np.array([pk for _, pk, _ in ps], int), spans=spans))
    import pickle
    with (out / "items.pkl").open("wb") as fh:
        pickle.dump(items, fh)
    meta = {s: sum(1 for i in items if i["split"] == s) for s in ("train", "test")}
    print("[prep]", meta, "ludb", sum(i["src"] == "ludb" for i in items), "qtdb", sum(i["src"] == "qtdb" for i in items))


def build_model():
    from torch import nn

    class Net(nn.Module):
        def __init__(self, c=24):
            super().__init__()
            layers, ch = [], 1
            for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16):
                layers += [nn.Conv1d(ch, c, 7, padding=3 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU()]
                ch = c
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(c, 1, 1)

        def forward(self, x):
            return self.head(self.body(x))
    return Net()


def train(data: Path, out: Path, epochs: int, seed: int) -> None:
    import pickle

    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    items = [i for i in pickle.load((data / "items.pkl").open("rb")) if i["split"] == "train"]
    # crops: LUDB anywhere in the annotated range; QTDB centred on annotated beats
    ludb = [i for i in items if i["src"] == "ludb"]
    qtdb = [i for i in items if i["src"] == "qtdb"]

    def crop():
        pool = qtdb if (qtdb and rng.random() < QTDB_SHARE) else ludb
        it = pool[rng.integers(len(pool))]
        n = it["x"].size
        if n <= WIN:
            return None
        if it["src"] == "qtdb" and it.get("spans"):
            a, _ = it["spans"][rng.integers(len(it["spans"]))]
            s = int(np.clip(a - rng.integers(0, WIN - 300), 0, n - WIN))
        else:
            s = int(rng.integers(0, n - WIN))
        x, y, m = it["x"][s:s + WIN], it["y"][s:s + WIN], it["m"][s:s + WIN]
        if m.sum() < 200:
            return None
        g = 1.0 + rng.normal(0, 0.15)
        x = x * (g if rng.random() < 0.5 else -g) if it["src"] == "qtdb" else x * g   # qtdb leads are unnamed
        x = x + rng.normal(0, 0.05, x.size).astype(np.float32)
        return x.astype(np.float32), y.astype(np.float32), m.astype(np.float32)
    net = build_model()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    steps = epochs * 200
    for st in range(steps):
        batch = []
        while len(batch) < 32:
            c = crop()
            if c is not None:
                batch.append(c)
        X = torch.tensor(np.stack([b[0] for b in batch]))[:, None]
        Y = torch.tensor(np.stack([b[1] for b in batch]))[:, None]
        M = torch.tensor(np.stack([b[2] for b in batch]))[:, None]
        loss = (bce(net(X), Y) * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 200 == 0:
            print(f"[train] step {st}/{steps} loss {loss.item():.4f}", flush=True)
        if st == int(steps * 0.7):
            for g in opt.param_groups:
                g["lr"] = 5e-4
    net.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 1, WIN), str(out), input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "b", 2: "t"}, "logit": {0: "b", 2: "t"}}, opset_version=17, dynamo=False)
    print("[train] saved", out, out.stat().st_size, "bytes")


def predict(sess, x: np.ndarray) -> np.ndarray:
    logit = sess.run(None, {"x": x[None, None, :].astype(np.float32)})[0][0, 0]
    return 1.0 / (1.0 + np.exp(-logit))


def peaks_from_prob(prob: np.ndarray, x: np.ndarray, thr: float = 0.5, min_len: int = 15) -> np.ndarray:
    on = prob >= thr
    edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int), [0]))))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            out.append((a + b) // 2)          # the centre of the P region (argmax |x| landed off the peak)
    return np.array(out, int)


def evaluate(data: Path, model: Path, thr: float, split: str = "test") -> dict:
    import pickle

    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    items = [i for i in pickle.load((data / "items.pkl").open("rb")) if i["split"] == split]
    tol = int(0.04 * FS)
    import csv
    rhythm = {int(r["ID"]): r["Rhythms"].strip() for r in csv.DictReader(open(ROOT / "data/ludb/ludb.csv"))}
    res = {}
    for lead in ("ii", "all"):     # false P on LUDB atrial fibrillation / flutter (no P at all)
        n = d_all = 0
        for it in items:
            if it["src"] == "ludb" and (lead == "all" or it["lead"] == lead) and rhythm[it["id"]].startswith("Atrial"):
                d = peaks_from_prob(predict(sess, it["x"]), it["x"], thr)
                d_all += int(((d >= it["lo"]) & (d <= it["hi"])).sum()); n += 1
        res[f"ludb_af_false_per_record:{lead}"] = round(d_all / max(1, n), 2)
    for src, lead in (("ludb", "ii"), ("ludb", "all"), ("qtdb", "all")):
        g = tp = fp = 0
        for it in items:
            if it["src"] != src or (lead != "all" and it["lead"] != lead):
                continue
            if src == "ludb" and rhythm[it["id"]].startswith("Atrial"):
                continue
            d = peaks_from_prob(predict(sess, it["x"]), it["x"], thr)
            if src == "ludb":
                d = d[(d >= it["lo"]) & (d <= it["hi"])]
                gold = it["peaks"]
                g += len(gold); tp += int(sum(np.any(np.abs(d - p) <= tol) for p in gold)) if len(d) else 0
                fp += int(sum(not np.any(np.abs(gold - p) <= tol) for p in d)) if len(gold) else len(d)
            else:   # only inside manually annotated beats
                inside = np.array([any(a <= p < b for a, b in it["spans"]) for p in d], bool) if len(d) else np.array([], bool)
                d = d[inside] if len(d) else d
                gold = it["peaks"]
                g += len(gold); tp += int(sum(np.any(np.abs(d - p) <= tol) for p in gold)) if len(d) else 0
                fp += int(sum(not np.any(np.abs(gold - p) <= tol) for p in d)) if len(gold) else len(d)
        res[f"{src}:{lead}"] = {"recall": round(tp / max(1, g), 3), "precision": round(tp / max(1, tp + fp), 3), "gold": g}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=295)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.out)
    elif a.cmd == "train":
        train(a.data, a.out, a.epochs, a.seed)
    else:
        print(json.dumps(evaluate(a.data, a.model, a.thr, a.split), indent=1))


if __name__ == "__main__":
    main()
