"""D294: per-beat P -> next-R spread (both signs) in II, aVF, V1 for data/avb_bench and the PTB-XL rhythm bench."""
import argparse, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))


def feats(res):
    import numpy as np
    from p_morph.pbeat import beat_p
    m, s = res.measurements, res.signal
    rp = np.asarray(m.r_peaks_samples or [], int)
    out = {"codes": sorted(res.findings.codes()), "hr": m.heart_rate_bpm, "qrs": m.intervals.qrs_ms}
    if rp.size >= 4:
        for ln in ("II", "aVF", "V1"):
            if ln in s.lead_names:
                b = beat_p(s.lead(ln), rp, s.sampling_rate_hz, m.intervals.qrs_ms, m.intervals.qt_ms)
                if b:
                    out[ln] = b
    return out


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["avb", "ptb"]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    if a.which == "avb":
        recs = json.loads((ROOT / "data/avb_bench/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/avb_bench" / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age")) for r in recs]
    else:
        recs = json.loads((ROOT / "data/ptbxl_rhythm/manifest.json").read_text())["records"]
        keep = {"SBRAD", "NORM", "1AVB", "3AVB", "2AVB", "SR"}
        items = [(str(ROOT / "data/ptbxl_rhythm" / r["filename_hr"]), r["ecg_id"], r["labels"], r.get("age"))
                 for r in recs if set(r["labels"]) & keep]
    with a.out.open("w") as fh:
        for path, ident, labels, age in items[k::n]:
            row = {"id": ident, "labels": labels}
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                row.update(feats(run_pipeline(sig, mode="A")))
            except Exception as ex:
                row["error"] = str(ex)[:60]
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
