"""S3 header-OCR bench (TZ 18.6, 18.14): engines compared on synthetic printed headers.

Per engine and header: character error rate of the whole header, speed and gain read right
(acceptance: >= 95 % of readable headers), each recorder field (measurements, date, model,
conclusion), and personal-data leaks: any token of the name, ID or birth date found in what the
scanner keeps (the safe lines and the parsed fields). A leak is a failure whatever the rest.

    python -m ecgscan.bench.header_bench --n 96 --out reports/scanner/s3-header-ocr

Engines: RapidOCR with its own recogniser, RapidOCR with the Cyrillic PP-OCRv5 recogniser
(``calibration.ocr.rec_model``), Tesseract rus+eng+uzb+uzb_cyrl when the binary is installed.
The RapidOCR engines run through ``read_bands`` as the pipeline does (tiles, colour retry).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ecgscan.bench.headers import HeaderCase, header_set
from ecgscan.calibration.header_fields import parse_fields
from ecgscan.calibration.header_ocr import group_lines, read_bands
from ecgscan.config import load_yaml, repo_root

FIELDS = ("hr", "pr", "qrs", "qt", "qtc", "axis", "recorded_at", "model", "conclusion")


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return " ".join(s.split())


_LOOK = str.maketrans("АВЕКМНОРСТХаеорсухТ", "ABEKMHOPCTXaeopcyxT")


def _same_text(a: str | None, b: str | None) -> bool:
    """Free text (model, conclusion) equal up to case, spaces, punctuation and Latin/Cyrillic
    look-alikes: what a reader would call the same words."""
    if a is None or b is None:
        return a is b
    def key(s: str) -> str:
        return "".join(c for c in s.translate(_LOOK).casefold() if c.isalnum())
    return key(a) == key(b)


def _rapid(rec_model: str | None, rec_sha256: str | None, ocr_cfg: dict, ppm: float):
    def run(img: np.ndarray) -> list[str]:
        r = read_bands(img, header_frac=1.0, footer_frac=0.0, min_band_px=int(4 * ppm),
                       min_tile_px=int(ocr_cfg.get("tile_min_mm", 0) * ppm),
                       overlap_px=int(ocr_cfg.get("tile_overlap_mm", 0) * ppm),
                       colour_frac=float(ocr_cfg.get("colour_retry_frac", 0.0)), rec_model=rec_model,
                       rec_sha256=rec_sha256, whole_header=True)
        if not r.available:
            raise RuntimeError(r.reason)
        return group_lines(r.boxes)
    return run


def _tesseract(img: np.ndarray) -> list[str]:
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, img)
        out = subprocess.run(["tesseract", f.name, "-", "-l", "rus+eng+uzb+uzb_cyrl", "--psm", "6"],
                             capture_output=True, text=True, timeout=120, check=False).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def engines(ppm: float) -> dict:
    ocr = load_yaml("pipeline")["calibration"]["ocr"]
    out = {"rapidocr_default": _rapid(None, None, ocr, ppm),
           "rapidocr_cyrillic_v5": _rapid(str(repo_root() / ocr["rec_model"]), ocr.get("rec_sha256"), ocr, ppm)}
    if shutil.which("tesseract"):
        out["tesseract_rus_eng_uzb"] = _tesseract
    return out


def _leaks(case: HeaderCase, kept: list[str], fields: dict) -> list[str]:
    """Personal-data tokens (name words, ID, birth date) found in what is kept."""
    phi = case.truth["phi"]
    tokens = [w for w in phi["name"].split() if len(w) >= 3] + [phi["id"], phi["dob"]]
    hay = " ".join(kept) + " " + json.dumps(fields, ensure_ascii=False)
    return [t for t in tokens if t.lower() in hay.lower()]


def score_case(case: HeaderCase, lines: list[str], models: list[str]) -> dict:
    t = case.truth
    truth_text = _norm(" ".join(ln.text for ln in case.lines))
    f = parse_fields(lines, models=models)
    m = f.measurements
    got = dict(hr=m.hr_bpm, pr=m.pr_ms, qrs=m.qrs_ms, qt=m.qt_ms, qtc=m.qtc_ms, axis=m.axis_deg,
               recorded_at=f.recorded_at, model=f.device_model, conclusion=f.conclusion)
    exp = dict(hr=t["hr"], pr=t["pr"], qrs=t["qrs"], qt=t["qt"], qtc=t["qtc"], axis=t["axis"],
               recorded_at=t["recorded_at"], model=t["model"], conclusion=t["conclusion"])
    kept = dict(got, speeds=f.speeds, gains=f.gains, hp=f.filter_hp_hz, lp=f.filter_lp_hz)
    return dict(
        case=case.case_id, style=case.style, degrade=json.dumps(case.degrade, sort_keys=True),
        cer=round(_lev(_norm(" ".join(lines)), truth_text) / max(1, len(truth_text)), 4),
        speed_ok=f.speeds[:1] == [t["speed"]] and len(set(f.speeds)) == 1,
        gain_ok=f.gains[:1] == [t["gain"]] and len(set(f.gains)) == 1,
        fields_ok={k: _same_text(got[k], exp[k]) if k in ("model", "conclusion") else got[k] == exp[k] for k in FIELDS},
        phi_dropped=f.phi_lines_dropped, leaks=_leaks(case, f.safe_lines, kept))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for eng in sorted({r["engine"] for r in rows}):
        rr = [r for r in rows if r["engine"] == eng]
        n = len(rr)
        s = dict(n=n, cer_median=round(float(np.median([r["cer"] for r in rr])), 4),
                 cer_mean=round(float(np.mean([r["cer"] for r in rr])), 4),
                 speed_ok=sum(r["speed_ok"] for r in rr), gain_ok=sum(r["gain_ok"] for r in rr),
                 speed_and_gain_ok_pct=round(100 * sum(r["speed_ok"] and r["gain_ok"] for r in rr) / n, 1),
                 fields_ok={k: sum(r["fields_ok"][k] for r in rr) for k in FIELDS},
                 cases_with_leak=sum(bool(r["leaks"]) for r in rr),
                 time_s_mean=round(float(np.mean([r["time_s"] for r in rr])), 2))
        s["by_style"] = {st: dict(n=sum(r["style"] == st for r in rr),
                                  speed_and_gain_ok=sum(r["speed_ok"] and r["gain_ok"] for r in rr if r["style"] == st),
                                  cer_median=round(float(np.median([r["cer"] for r in rr if r["style"] == st])), 4))
                         for st in ("ru", "en", "uz") if any(r["style"] == st for r in rr)}
        s["by_degrade"] = {d: sum(r["speed_ok"] and r["gain_ok"] for r in rr if r["degrade"] == d)
                           for d in sorted({r["degrade"] for r in rr})}
        out[eng] = s
    return out


def run(n: int, seed: int, out: Path, ppm: float = 10.0, only: list[str] | None = None) -> dict:
    models = load_yaml("pipeline")["calibration"]["ocr"].get("device_models", [])
    cases = header_set(n, seed=seed, px_per_mm=ppm)
    rows = []
    for name, eng in engines(ppm).items():
        if only and name not in only:
            continue
        for c in cases:
            t0 = time.perf_counter()
            lines = eng(c.image)
            row = score_case(c, lines, models)
            row.update(engine=name, time_s=round(time.perf_counter() - t0, 2))
            rows.append(row)
    summary = dict(n_cases=n, seed=seed, px_per_mm=ppm, engines=summarize(rows))
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                     encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="header OCR engines on synthetic printed headers (S3)")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", nargs="*")
    a = ap.parse_args(argv)
    s = run(a.n, a.seed, a.out, only=a.engines)
    print(json.dumps(s, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
