"""Train the M6 trace segmenter and export it to ONNX (SD-051, SD-052).

A small U-Net (4 levels, widths ``--channels``) trained on CPU from the
shards of ``build_segmenter_data.py``. PyTorch is a training-only dependency (kept
out of the runtime: the service runs the exported model with onnxruntime).

    python tools/scanner/train_segmenter.py --data data/segmenter --out models/segmenter --iters 4000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Input: N×3×H×W BGR scaled to [-0.5, 0.5], H and W multiples of 8. Output: stroke logits N×1×H×W."""

    def __init__(self, ch: tuple[int, ...] = (16, 32, 64, 128)):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 3
        for c in ch:
            self.enc.append(block(cin, c))
            cin = c
        self.dec = nn.ModuleList(block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1)))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for dec, skip in zip(self.dec, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def load(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in files:
        z = np.load(f)
        xs.append(z["x"])
        ys.append(z["y"])
    return np.concatenate(xs), np.concatenate(ys)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255.0).sub(0.5)


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x, y = x.copy(), y.copy()
    for i in range(len(x)):
        if rng.random() < 0.5:
            x[i], y[i] = x[i, :, ::-1], y[i, :, ::-1]
        if rng.random() < 0.3:
            x[i], y[i] = x[i, ::-1], y[i, ::-1]
        if rng.random() < 0.3:                                  # another grid hue
            x[i] = x[i][..., rng.permutation(3)]
        if rng.random() < 0.5:                                  # exposure / contrast
            a, b = rng.uniform(0.7, 1.2), rng.uniform(-30, 20)
            x[i] = np.clip(x[i].astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return x, y


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(4.0))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, batch: int = 32) -> dict:
    model.eval()
    tp = fp = fn = 0
    for i in range(0, len(x), batch):
        p = torch.sigmoid(model(to_tensor(x[i:i + batch]))).squeeze(1).numpy() > 0.5
        t = y[i:i + batch].astype(bool)
        tp += int((p & t).sum())
        fp += int((p & ~t).sum())
        fn += int((~p & t).sum())
    model.train()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4)}


def export(model: nn.Module, path: Path) -> str:
    model.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}}, dynamo=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/segmenter")
    ap.add_argument("--out", default="models/segmenter")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", default="16,32,64,128", help="U-Net widths per level (narrower = faster inference)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    data = Path(a.data)
    xt, yt = load(sorted(data.glob("train_*.npz")))
    xv, yv = load(sorted(data.glob("val_*.npz")))
    print(f"train {len(xt)} crops, val {len(xv)} crops, ink {yt.mean():.3f}", flush=True)
    model = UNet(tuple(int(c) for c in a.channels.split(",")))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history, t0 = -1.0, [], time.time()
    for it in range(1, a.iters + 1):
        idx = rng.integers(0, len(xt), a.batch)
        xb, yb = augment(xt[idx], yt[idx], rng)
        loss = loss_fn(model(to_tensor(xb)), torch.from_numpy(yb.astype(np.float32)).unsqueeze(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % a.eval_every == 0 or it == a.iters:
            m = evaluate(model, xv, yv)
            history.append({"iter": it, "loss": round(float(loss), 4), **m, "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(history[-1]), flush=True)
            if m["f1"] > best:
                best = m["f1"]
                torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    sha = export(model, out / "trace_unet.onnx")
    card = {"model": "trace_unet.onnx", "sha256": sha, "params": n_params, "channels": a.channels, "input": "N×3×H×W BGR/255-0.5, H,W % 8 == 0",
            "output": "stroke logits N×1×H×W", "train_crops": int(len(xt)), "val_crops": int(len(xv)),
            "iters": a.iters, "batch": a.batch, "lr": a.lr, "best_val": max(history, key=lambda h: h["f1"]),
            "history": history, "data": "synthetic only (ecgscan.segmentation.dataset, SEED_BASE range)"}
    (out / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print("exported", out / "trace_unet.onnx", sha, flush=True)


if __name__ == "__main__":
    main()
