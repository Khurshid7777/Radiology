"""D282-D284: one pipeline pass per PTB-XL record for the descriptive wave-5 findings — electrical alternans, short PR
without pre-excitation, P polarity (ectopic atrial rhythm). Rows hold numbers and codes only.

    python tools/wave5/run.py <data dir with manifest.json> --out rows.jsonl [--shard k/n]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def row_for(sig) -> dict:
    from cardiolens.arrhythmia.alternans import detect_alternans
    from cardiolens.pipeline import run_pipeline
    from p_morph.pbeat import beat_p

    res = run_pipeline(sig, mode="A")
    m, s = res.measurements, res.signal
    rp = np.asarray(m.r_peaks_samples or [], int)
    rr = np.diff(rp)
    alt, det = detect_alternans(s.signal, list(s.lead_names), m.r_peaks_samples or [], s.sampling_rate_hz)
    row = {"codes": sorted(res.findings.codes()), "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms,
           "pr": m.intervals.pr_ms, "pon": m.p_onset_ratio, "prstd": m.pr_std_ms,
           "cv": float(rr.std() / rr.mean()) if rr.size > 2 else None,
           "alt": alt, "alt_hits": det.get("hits", []), "alt_rr_flip": det.get("rr_flip"),
           "p_axis": getattr(getattr(m, "axes", None), "p_deg", None)}
    if rp.size >= 4:
        for ln in ("II", "aVF", "III", "I", "aVR"):
            if ln in s.lead_names:
                b = beat_p(s.lead(ln), rp, s.sampling_rate_hz, m.intervals.qrs_ms, m.intervals.qt_ms)
                if b:
                    row["P_" + ln] = {k: b[k] for k in ("n", "pos", "neg", "pos_pr", "neg_pr")}
    return row


def main() -> None:
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    recs = json.loads((a.data / "manifest.json").read_text())["records"][k::n]
    with a.out.open("w") as fh:
        for r in recs:
            row = {"id": r["ecg_id"], "labels": r["labels"]}
            try:
                row.update(row_for(load_wfdb(str(a.data / r["filename_hr"]))))
            except Exception as ex:
                row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
