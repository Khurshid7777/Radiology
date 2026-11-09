"""D219: small ECG pictures with a known signal — the score that the picture preparation is tuned on.

    python tools/lowres_bench/run.py gen   --kit <ecg-image-generator> --kit-python <py> --n 80 --out <dir>
    python tools/lowres_bench/run.py score --out <dir> [--prep '{"enabled": true, "factor": 2}'] [--half even|odd|all]

gen: PTB-XL records -> ECG-Image-Kit pages at 300 dpi (a 12-lead sheet ~2950 px wide, ~12 px/mm), then each
page shrunk to widths 1000 / 1300 / 1600 px and saved as JPEG quality 80 (messenger / atlas-like), plus the
truth (the pipeline on the record itself). score: every small picture through the site's photo route
with the given preparation (cardiolens.layer1_ingestion.image_prep.PREP), and

    score = R x A - 3 x W

R read share (no refusal, no photo gate), A accurate share of the read (HR within 5 bpm and QRS within 20 ms
of the record), W silent-wrong share of all (read without a gate, HR off by more than 10 %). Images and rows
stay in --out (outside git); only the summary goes to reports/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "eik_bench"))
WIDTHS = (1000, 1300, 1600)
GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible", "photo_coarse"}


def gen(a) -> None:
    from PIL import Image
    from run import generate, pick_records  # tools/eik_bench/run.py

    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline

    inp, img, small = a.out / "records", a.out / "pages", a.out / "small"
    for d in (inp, img, small):
        d.mkdir(parents=True, exist_ok=True)
    for hea in pick_records(a.ptbxl, a.n, a.seed):
        for ext in (".hea", ".dat"):
            shutil.copy(hea.with_suffix(ext), inp)
    generate(a.kit, a.kit_python, inp, img, a.seed, [])
    truth = {}
    for png in sorted(img.glob("*-0.png")):
        rec = png.name.rsplit("-", 1)[0]
        m = run_pipeline(load_wfdb(str(inp / rec))).measurements
        truth[rec] = {"hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}
        im = Image.open(png).convert("RGB")
        for w in WIDTHS:
            im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(small / f"{rec}_w{w}.jpg", quality=80)
    (a.out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(len(truth), "records,", len(truth) * len(WIDTHS), "small pictures", file=sys.stderr)


def _one(args) -> dict:
    path, prep = args
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import image_prep
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    image_prep.PREP.update(prep)
    try:
        sig = digitize_photos([path])
        res = run_pipeline(sig, mode="A")
    except Exception as e:  # a refusal is a result
        return {"img": Path(path).name, "read": False, "why": str(e)[:80]}
    m = res.measurements
    return {"img": Path(path).name, "read": True, "gated": bool(res.findings.codes() & GATES),
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}


def score_rows(rows: list[dict], truth: dict) -> dict:
    n = len(rows) or 1
    ok_read = [r for r in rows if r["read"] and not r["gated"]]
    def rec(r):
        return r["img"].rsplit("_w", 1)[0]
    def acc(r):
        t = truth[rec(r)]
        return (r["hr"] is not None and t["hr"] and abs(r["hr"] - t["hr"]) <= 5
                and r["qrs"] is not None and t["qrs"] and abs(r["qrs"] - t["qrs"]) <= 20)
    def wrong(r):
        t = truth[rec(r)]
        return r["hr"] is not None and t["hr"] and abs(r["hr"] - t["hr"]) > 0.10 * t["hr"]
    R = len(ok_read) / n
    A = sum(acc(r) for r in ok_read) / max(1, len(ok_read))
    W = sum(wrong(r) for r in ok_read) / n
    return {"n": len(rows), "R": round(R, 3), "A": round(A, 3), "W": round(W, 3), "score": round(R * A - 3 * W, 3)}


def score(a) -> None:
    truth = json.loads((a.out / "truth.json").read_text())
    prep = json.loads(a.prep)
    imgs = sorted((a.out / "small").glob("*.jpg"))
    if a.half != "all":
        want = 0 if a.half == "even" else 1
        imgs = [p for p in imgs if int(p.name.split("_")[0]) % 2 == want]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, [(str(p), prep) for p in imgs]))
    out = {"prep": prep, "half": a.half, "all": score_rows(rows, truth)}
    for w in WIDTHS:
        out[f"w{w}"] = score_rows([r for r in rows if r["img"].endswith(f"_w{w}.jpg")], truth)
    print(json.dumps(out, ensure_ascii=False))
    if a.rows:
        a.rows.write_text("\n".join(json.dumps(r) for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("gen", "score"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kit", type=Path)
    ap.add_argument("--kit-python")
    ap.add_argument("--ptbxl", type=Path, default=ROOT / "data/ptbxl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prep", default='{"enabled": false}')
    ap.add_argument("--half", default="all", choices=("even", "odd", "all"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rows", type=Path)
    a = ap.parse_args()
    gen(a) if a.step == "gen" else score(a)


if __name__ == "__main__":
    main()
