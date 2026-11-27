"""D366: P detector with a second input channel — the lead minus the record's median beat (template_subtract).

    python tools/p_detector/cnn_resid.py prep  --out <dir>        # items + R peaks from QRS annotations
    python tools/p_detector/cnn_resid.py train --data <dir> --out <model.onnx>
    python tools/p_detector/cnn_resid.py eval  --data <dir> --model <model.onnx> --split val|test

Split as D295 (val = LUDB id % 4 == 0 and every 4th even QTDB; test = odd). Inserted P go into the whole 10-s segment
BEFORE the median beat is subtracted, at random places relative to QRS. Env: PDET_CH (24), PDET_STEPS (5000),
PDET_INSERT (0.5), PDET_AF_SHARE (0.15), PDET_QTDB_SHARE (0.4).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn import FS, WIN, insert_p, p_bank, peaks_from_prob, prep as prep_v1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cardiolens.layer3_measure.atrial import template_subtract  # noqa: E402

PRE_S, POST_S = 0.30, 0.55
BEATS = set("NLRAaJSVFejE/fQ")
SEG = 5000          # 10 s at 500 Hz — the residual is computed on a 10-s segment, as in a real ECG
CH = int(os.environ.get("PDET_CH", "24"))
STEPS = int(os.environ.get("PDET_STEPS", "5000"))
INSERT = float(os.environ.get("PDET_INSERT", "0.5"))
AF_SHARE = float(os.environ.get("PDET_AF_SHARE", "0.15"))
QTDB_SHARE = float(os.environ.get("PDET_QTDB_SHARE", "0.4"))


def residual(x: np.ndarray, rp: np.ndarray) -> np.ndarray:
    rp = rp[(rp >= 0) & (rp < x.size)]
    r = template_subtract(x, rp, FS, PRE_S, POST_S) if rp.size >= 3 else None
    return (x if r is None else r).astype(np.float32)


def two(x: np.ndarray, rp: np.ndarray) -> np.ndarray:
    return np.stack([x.astype(np.float32), residual(x, rp)])


def _qrs(ann, scale: float = 1.0) -> np.ndarray:
    return np.array([int(s * scale) for s, y in zip(ann.sample, ann.symbol) if y in BEATS], int)


def prep(out: Path) -> None:
    # items.pkl / items2.pkl are written by this tool from local open datasets (numbers only) — not untrusted input
    import wfdb
    prep_v1(out)
    items = pickle.load((out / "items.pkl").open("rb"))
    cache: dict = {}
    for it in items:
        if it["src"] == "ludb":
            ann = wfdb.rdann(str(ROOT / "data/ludb/data" / str(it["id"])), it["lead"])
            it["rp"] = _qrs(ann)
        else:
            if it["id"] not in cache:
                p = ROOT / "data/qtdb" / it["id"]
                rec = wfdb.rdrecord(str(p))
                sc = FS / rec.fs
                cache[it["id"]] = _qrs(wfdb.rdann(str(p), "atr"), sc)
            it["rp"] = cache[it["id"]]
    import csv
    rhythm = {int(r["ID"]): r["Rhythms"].strip() for r in csv.DictReader(open(ROOT / "data/ludb/ludb.csv"))}
    for it in items:
        it["af"] = it["src"] == "ludb" and rhythm[it["id"]].startswith("Atrial")
    with (out / "items2.pkl").open("wb") as fh:
        pickle.dump(items, fh)
    print("[prep2] items", len(items), "with R", sum(len(i["rp"]) > 2 for i in items))


def segment(it, rng):
    """A 10-s segment (x, y, m, rp) — LUDB whole record; QTDB around an annotated beat."""
    n = it["x"].size
    if it["src"] == "qtdb":
        a, _ = it["spans"][rng.integers(len(it["spans"]))]
        s = int(np.clip(a - rng.integers(500, SEG - 500), 0, max(0, n - SEG)))
    else:
        s = 0
    e = min(n, s + SEG)
    rp = it["rp"][(it["rp"] >= s) & (it["rp"] < e)] - s
    return it["x"][s:e].copy(), it["y"][s:e].copy(), it["m"][s:e].copy(), rp


def build_model():
    from torch import nn

    class Net(nn.Module):
        def __init__(self, c=CH):
            super().__init__()
            layers, ch = [], 2
            for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16):
                layers += [nn.Conv1d(ch, c, 7, padding=3 * d, dilation=d), nn.BatchNorm1d(c), nn.ReLU()]
                ch = c
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(c, 1, 1)

        def forward(self, x):
            return self.head(self.body(x))
    return Net()


def train(data: Path, out: Path, seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    rng = np.random.default_rng(seed)
    items = [i for i in pickle.load((data / "items2.pkl").open("rb")) if i["split"] == "train"]
    ludb = [i for i in items if i["src"] == "ludb" and not i["af"]]
    af = [i for i in items if i["af"]]
    qtdb = [i for i in items if i["src"] == "qtdb" and i.get("spans")]
    bank = p_bank(ludb, rng)

    def crop():
        u = rng.random()
        pool = af if u < AF_SHARE else (qtdb if u < AF_SHARE + QTDB_SHARE * (1 - AF_SHARE) else ludb)
        it = pool[rng.integers(len(pool))]
        x, y, m, rp = segment(it, rng)
        if x.size <= WIN:
            return None
        g = 1.0 + rng.normal(0, 0.15)
        x = x * (g if (it["src"] != "qtdb" or rng.random() < 0.5) else -g)
        if not it["af"] and rng.random() < INSERT:
            lo = max(0, int(np.flatnonzero(m)[0])) if m.any() else 0
            hi = int(np.flatnonzero(m)[-1]) if m.any() else x.size
            sx, sy = x[lo:hi], y[lo:hi]
            insert_p(sx, sy, bank, rng, int(rng.integers(2, 7)))
        x = x + rng.normal(0, 0.05, x.size).astype(np.float32)
        X = two(x, rp)
        s = int(rng.integers(0, x.size - WIN))
        mm = m[s:s + WIN]
        if mm.sum() < 200:
            return None
        return X[:, s:s + WIN], y[s:s + WIN].astype(np.float32), mm.astype(np.float32)
    net = build_model()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    for st in range(STEPS):
        batch = []
        while len(batch) < 32:
            c = crop()
            if c is not None:
                batch.append(c)
        X = torch.tensor(np.stack([b[0] for b in batch]))
        Y = torch.tensor(np.stack([b[1] for b in batch]))[:, None]
        M = torch.tensor(np.stack([b[2] for b in batch]))[:, None]
        loss = (bce(net(X), Y) * M).sum() / M.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if st % 250 == 0:
            print(f"[train] step {st}/{STEPS} loss {loss.item():.4f}", flush=True)
        if st == int(STEPS * 0.7):
            for g in opt.param_groups:
                g["lr"] = 5e-4
    net.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.zeros(1, 2, WIN), str(out), input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "b", 2: "t"}, "logit": {0: "b", 2: "t"}}, opset_version=17, dynamo=False)
    print("[train] saved", out, out.stat().st_size, "bytes")


def predict(sess, X: np.ndarray) -> np.ndarray:
    logit = sess.run(None, {"x": X[None].astype(np.float32)})[0][0, 0]
    return 1.0 / (1.0 + np.exp(-logit))


def _score(d, gold, tol):
    tp = int(sum(np.any(np.abs(d - p) <= tol) for p in gold)) if len(d) else 0
    fp = int(sum(not np.any(np.abs(gold - p) <= tol) for p in d)) if len(gold) else len(d)
    return tp, fp


def evaluate(data: Path, model: Path, thr: float, split: str, seed: int) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    allit = pickle.load((data / "items2.pkl").open("rb"))
    items = [i for i in allit if i["split"] == split]
    tol = int(0.04 * FS)
    res: dict = {}
    n = fa = 0
    for it in items:
        if it["af"] and it["lead"] == "ii":
            d = peaks_from_prob(predict(sess, two(it["x"], it["rp"])), it["x"], thr)
            fa += int(((d >= it["lo"]) & (d <= it["hi"])).sum())
            n += 1
    res["ludb_af_false_per_record:ii"] = round(fa / max(1, n), 2)
    for src, lead in (("ludb", "ii"), ("ludb", "all"), ("qtdb", "all")):
        g = tp = fp = 0
        for it in items:
            if it["src"] != src or (lead != "all" and it["lead"] != lead) or it["af"]:
                continue
            if src == "ludb":
                d = peaks_from_prob(predict(sess, two(it["x"], it["rp"])), it["x"], thr)
                d = d[(d >= it["lo"]) & (d <= it["hi"])]
            else:
                if not it.get("spans"):
                    continue
                d = []
                for a, b in it["spans"][::max(1, len(it["spans"]) // 40)]:
                    s = max(0, a - SEG // 2)
                    e = min(it["x"].size, s + SEG)
                    rp = it["rp"][(it["rp"] >= s) & (it["rp"] < e)] - s
                    dd = peaks_from_prob(predict(sess, two(it["x"][s:e], rp)), it["x"][s:e], thr) + s
                    gold_b = it["peaks"][(it["peaks"] >= a) & (it["peaks"] < b)]
                    dd = dd[(dd >= a) & (dd < b)]
                    t_, f_ = _score(dd, gold_b, tol)
                    g += len(gold_b)
                    tp += t_
                    fp += f_
                continue
            gold = it["peaks"]
            t_, f_ = _score(d, gold, tol)
            g += len(gold)
            tp += t_
            fp += f_
        res[f"{src}:{lead}"] = {"recall": round(tp / max(1, g), 3), "precision": round(tp / max(1, tp + fp), 3), "gold": g}
    # inserted P, placed before the median beat is subtracted
    rng = np.random.default_rng(seed)
    bank = p_bank([i for i in allit if i["split"] == "train" and i["src"] == "ludb" and not i["af"]], rng)
    found = total = 0
    for it in items:
        if it["src"] != "ludb" or it["lead"] != "ii" or it["af"]:
            continue
        x, y = it["x"].copy(), it["y"].copy()
        before = y.copy()
        lo, hi = it["lo"], it["hi"]
        sx, sy = x[lo:hi], y[lo:hi]
        insert_p(sx, sy, bank, rng, 4)
        new = np.flatnonzero(np.diff(np.concatenate(([0], (y[lo:hi] & ~before[lo:hi]).astype(int), [0]))))
        centres = [lo + (a + b) // 2 for a, b in zip(new[::2], new[1::2])]
        d = peaks_from_prob(predict(sess, two(x, it["rp"])), x, thr)
        total += len(centres)
        found += sum(bool(len(d)) and bool(np.any(np.abs(d - c) <= tol)) for c in centres)
    res["inserted"] = {"n": total, "recall": round(found / max(1, total), 3)}
    for rid in (104, 111):    # LUDB complete AV block, described only
        for it in items:
            if it["src"] == "ludb" and it["id"] == rid and it["lead"] == "ii":
                d = peaks_from_prob(predict(sess, two(it["x"], it["rp"])), it["x"], thr)
                d = d[(d >= it["lo"]) & (d <= it["hi"])]
                t_, f_ = _score(d, it["peaks"], tol)
                res[f"ludb_3avb_{rid}"] = {"gold": len(it["peaks"]), "tp": t_, "fp": f_}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--seed", type=int, default=366)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.out)
    elif a.cmd == "train":
        train(a.data, a.out, a.seed)
    else:
        print(json.dumps(evaluate(a.data, a.model, a.thr, a.split, 777 if a.split == "val" else 991), indent=1))


if __name__ == "__main__":
    main()
