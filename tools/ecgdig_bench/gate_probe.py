"""D189 probe: why each ecgdig image is gated, and how good its rhythm lead looks without the
truth (beats against the page columns, QRS template consistency, beat count). One JSON per
image in <out>/probe_rows, resumable; the truth rate is kept to score a rate-only release.

    python tools/ecgdig_bench/gate_probe.py --data <dir with train.csv and train/> --out <dir> [--shard 0/3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import truth_signal


def probe(sig, res) -> dict:
    m = res.measurements
    f = next((f for f in res.findings.findings if f.code == "photo_unreliable"), None)
    ev = {e.feature: e.value for e in (f.evidence if f else [])}
    notes = sig.provenance.notes or []
    warn = next((n[10:] for n in notes if n.startswith("warnings: ")), "")
    scanner = next((n for n in notes if n.startswith("ecgscan scan_id=")), "")
    return {"route": "scanner" if scanner else "readers",
            "status": scanner.split("status=")[1].split()[0] if "status=" in scanner else None,
            "warnings": warn.replace(",", " ").split(),
            "gated": f is not None, "why": f.label_ru.split("(")[1].split(")")[0] if f and "(" in f.label_ru else None,
            "einthoven_r": ev.get("einthoven_r"), "panel_lag_ms": ev.get("panel_lag_ms"),
            "notes": [n[:60] for n in notes if n.startswith(("leads inconsistent", "grid in trace", "scale conflict"))],
            "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms, "agree": m.page_beat_agreement,
            "tc": m.qrs_template_consistency, "n_beats": len(m.r_peaks_samples or []),
            "duration_s": sig.duration_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    from cardiolens.pipeline import run_pipeline
    from cardiolens.telegram.analyze import digitize_photos

    fs_of = {r["id"]: float(r["fs"]) for r in csv.DictReader((a.data / "train.csv").open())}
    recs = sorted(p.name for p in (a.data / "train").iterdir() if p.is_dir())
    k, n = (int(v) for v in a.shard.split("/"))
    rows = a.out / "probe_rows"
    rows.mkdir(parents=True, exist_ok=True)
    for rec in recs[k::n]:
        truth, _ = truth_signal(a.data / "train" / rec / f"{rec}.csv", fs_of[rec])
        mt = run_pipeline(truth, mode="A").measurements
        for img in sorted((a.data / "train" / rec).glob(f"{rec}-v*.*")):
            cond = img.stem.split("-")[1]
            out = rows / f"{rec}-{cond}.json"
            if out.exists():
                continue
            row = {"record": rec, "cond": cond, "hr_true": mt.heart_rate_bpm, "qrs_true": mt.intervals.qrs_ms}
            try:
                sig = digitize_photos([img])
                row.update(read=True, **probe(sig, run_pipeline(sig, mode="A")))
            except Exception as e:
                row.update(read=False, error=str(e)[:120])
            out.write_text(json.dumps(row, ensure_ascii=False))
            print(rec, cond, row.get("gated"), row.get("why"), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
