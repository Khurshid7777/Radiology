"""ECG-Image-Kit bench (D181): realistic photos of PTB-XL records through the site's photo route.

ECG-Image-Kit (github.com/alphanumericslab/ecg-image-kit) renders a WFDB record as a paper
ECG with creases, wrinkles, shadows, noise and rotation; the record itself is the ground
truth. Each image goes through ``digitize_photos`` + ``run_pipeline`` exactly as an upload
on the site, and is compared with the pipeline on the source record.

    python tools/eik_bench/run.py --kit <ecg-image-generator dir> --kit-python <py3.11 venv python> \
        --n 100 --seed 1 --out <dir outside git>

Images and per-case results stay in --out (never in git); the summary JSON can be copied
to reports/. Metrics (D181): read rate, HR within 5 bpm, QRS within 20 ms, and the
"silent wrong" rate — HR off by more than 10 % on a record not marked unreliable.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

GATES = {"photo_unreliable", "no_ecg_signal", "amplitude_implausible", "photo_coarse"}


def pick_records(ptbxl: Path, n: int, seed: int) -> list[Path]:
    heas = sorted(ptbxl.glob("records500/*/*.hea"))
    random.Random(seed).shuffle(heas)
    return heas[:n]


def generate(kit: Path, kit_py: str, inp: Path, out: Path, seed: int, extra: list[str]) -> None:
    """One generator call per record: a crash on one record must not stop the batch."""
    for hea in sorted(inp.glob("*.hea")):
        if (out / f"{hea.stem}-0.png").exists():
            continue
        one = out / "_one"
        shutil.rmtree(one, ignore_errors=True)
        one.mkdir()
        for ext in (".hea", ".dat"):
            shutil.copy(hea.with_suffix(ext), one)
        cmd = [kit_py, "gen_ecg_images_from_data_batch.py", "-i", str(one), "-o", str(out), "-se", str(seed),
               "-r", "300", "--print_header", "--store_config", "2", *extra]
        r = subprocess.run(cmd, cwd=kit, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            print(f"generator failed on {hea.stem}: {r.stderr.strip().splitlines()[-1:]}", file=sys.stderr)


def evaluate(out: Path, inp: Path) -> list[dict]:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    rows = []
    for png in sorted(out.glob("*-0.png")):
        rec = png.name.rsplit("-", 1)[0]
        truth = run_pipeline(load_wfdb(str(inp / rec))).measurements
        row = {"record": rec, "hr_true": truth.heart_rate_bpm, "qrs_true": truth.intervals.qrs_ms}
        try:
            sig = digitize_photos([png])
            res = run_pipeline(sig)
            m = res.measurements
            row.update(read=True, hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms,
                       route="scanner" if sig.provenance.notes[0].startswith("ecgscan") else "readers",
                       gated=bool(res.findings.codes() & GATES))
        except Exception as e:  # a refusal is a result too
            row.update(read=False, error=str(e)[:200])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    read = [r for r in rows if r.get("read")]
    def ok_hr(r):
        return r["hr"] is not None and r["hr_true"] and abs(r["hr"] - r["hr_true"]) <= 5
    def ok_qrs(r):
        return r["qrs"] is not None and r["qrs_true"] and abs(r["qrs"] - r["qrs_true"]) <= 20
    silent = [r for r in read if not r["gated"] and r["hr"] and r["hr_true"]
              and abs(r["hr"] - r["hr_true"]) > 0.10 * r["hr_true"]]
    n = len(rows) or 1
    return {"n": len(rows), "read": len(read), "read_rate": round(len(read) / n, 3),
            "hr_ok_of_all": round(sum(ok_hr(r) for r in read) / n, 3),
            "qrs_ok_of_all": round(sum(ok_qrs(r) for r in read) / n, 3),
            "gated": sum(r["gated"] for r in read),
            "silent_wrong_hr": len(silent), "silent_wrong_rate": round(len(silent) / n, 3),
            "by_route": {k: sum(1 for r in read if r["route"] == k) for k in ("scanner", "readers")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--kit-python", required=True)
    ap.add_argument("--ptbxl", default="data/ptbxl", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--distortions", default="--wrinkles --augment -rot 4")
    a = ap.parse_args()
    inp, img = a.out / "records", a.out / "images"
    for d in (inp, img):
        d.mkdir(parents=True, exist_ok=True)
    for hea in pick_records(a.ptbxl, a.n, a.seed):
        for ext in (".hea", ".dat"):
            shutil.copy(hea.with_suffix(ext), inp)
    generate(a.kit, a.kit_python, inp, img, a.seed, a.distortions.split())
    rows = evaluate(img, inp)
    summary = {"seed": a.seed, "distortions": a.distortions, **summarize(rows)}
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
