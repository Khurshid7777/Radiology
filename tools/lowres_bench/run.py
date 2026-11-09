"""D219: small ECG pictures with a known signal — the score that the picture preparation is tuned on.

    python tools/lowres_bench/run.py gen   --out <dir> [--n 80]   # pages drawn by ecgscan.bench.render
    python tools/lowres_bench/run.py score --out <dir> [--prep '{"enabled": true, "factor": 2}'] [--half even|odd|all]

gen: PTB-XL records -> paper pages drawn at 12 px/mm by the scanner's bench renderer (used read-only;
layouts 3x4+rhythm and 6x2 alternate, pink grid, printed header with 25 mm/s 10 mm/mV), then each page shrunk to widths 1000 / 1300 / 1600 px and saved as JPEG quality 80 (messenger / atlas-like), plus the
truth (the pipeline on the record itself). score: every small picture through the site's photo route
with the given preparation (cardiolens.layer1_ingestion.image_prep.PREP), and

    score = R_full x A_full + 0.5 x R_coarse x A_hr - 3 x W     (see score_rows) Images and rows
stay in --out (outside git); only the summary goes to reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
WIDTHS = (1000, 1300, 1600)
GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible"}


def gen(a) -> None:
    import random

    import cv2

    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    from ecgscan.bench.render import RenderSpec, render_page
    from ecgscan.layout.registry import get_layout

    small = a.out / "small"
    small.mkdir(parents=True, exist_ok=True)
    heas = sorted(a.ptbxl.glob("records500/*/*.hea"))
    random.Random(a.seed).shuffle(heas)
    truth = {}
    for k, hea in enumerate(heas[: a.n]):
        rec = hea.stem
        sig = load_wfdb(str(hea.with_suffix("")))
        m = run_pipeline(sig).measurements
        truth[rec] = {"hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}
        mv = {ld: sig.lead(ld).astype(float) / 1000.0 for ld in sig.lead_names}   # uV -> mV
        layout = get_layout("3x4+rhythm" if k % 2 == 0 else "6x2")
        img, _ = render_page(mv, sig.sampling_rate_hz, layout, RenderSpec(px_per_mm=12.0, grid="pink"))
        for w in WIDTHS:
            h = round(img.shape[0] * w / img.shape[1])
            cv2.imwrite(str(small / f"{rec}_w{w}.jpg"), cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
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
    codes = res.findings.codes()
    return {"img": Path(path).name, "read": True, "gated": bool(codes & GATES), "coarse": "photo_coarse" in codes,
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}


def score_rows(rows: list[dict], truth: dict) -> dict:
    """score = R_full x A_full + 0.5 x R_coarse x A_hr - 3 x W (D219).

    full: read without a gate; A_full = HR within 5 bpm and QRS within 20 ms. coarse: the scanner's coarse
    read (photo_coarse: rate and rhythm given as approximate, intervals withheld); A_hr = HR within 5 bpm.
    W (of all): read without photo_unreliable/no-signal, HR off by more than 10 %."""
    n = len(rows) or 1
    def rec(r):
        return r["img"].rsplit("_w", 1)[0]
    def hr_ok(r):
        t = truth[rec(r)]["hr"]
        return r.get("hr") is not None and t and abs(r["hr"] - t) <= 5
    def qrs_ok(r):
        t = truth[rec(r)]["qrs"]
        return r.get("qrs") is not None and t and abs(r["qrs"] - t) <= 20
    def wrong(r):
        t = truth[rec(r)]["hr"]
        return r.get("hr") is not None and t and abs(r["hr"] - t) > 0.10 * t
    full = [r for r in rows if r["read"] and not r.get("gated") and not r.get("coarse")]
    coarse = [r for r in rows if r["read"] and r.get("coarse") and not r.get("gated")]
    R_f, R_c = len(full) / n, len(coarse) / n
    A_f = sum(hr_ok(r) and qrs_ok(r) for r in full) / max(1, len(full))
    A_hr = sum(hr_ok(r) for r in coarse) / max(1, len(coarse))
    W = sum(wrong(r) for r in full + coarse) / n
    return {"n": len(rows), "R_full": round(R_f, 3), "A_full": round(A_f, 3), "R_coarse": round(R_c, 3),
            "A_hr_coarse": round(A_hr, 3), "W": round(W, 3), "score": round(R_f * A_f + 0.5 * R_c * A_hr - 3 * W, 3)}


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
