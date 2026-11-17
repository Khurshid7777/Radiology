"""D296: P-CNN dissociation features over data/avb_bench and the PTB-XL rhythm bench (numbers and codes only)."""
import argparse, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    warnings.filterwarnings("ignore")
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer3_measure.p_cnn import dissociation_features, p_peaks
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["avb", "ptb"]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    if a.which == "avb":
        recs = json.loads((ROOT / "data/avb_bench/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/avb_bench" / r["folder"] / r["name"]), r["name"], r["labels"], r.get("age")) for r in recs]
    else:
        recs = json.loads((ROOT / "data/ptbxl_rhythm/manifest.json").read_text())["records"]
        items = [(str(ROOT / "data/ptbxl_rhythm" / r["filename_hr"]), r["ecg_id"], r["labels"], r.get("age")) for r in recs]
    with a.out.open("w") as fh:
        for path, ident, labels, age in items[k::n]:
            row = {"id": ident, "labels": labels}
            try:
                sig = load_wfdb(path)
                try:
                    sig.patient.age_years = float(age) if age not in (None, "") else None
                except (TypeError, ValueError):
                    pass
                res = run_pipeline(sig, mode="A"); m, s = res.measurements, res.signal
                row.update(codes=sorted(res.findings.codes()), alerts=sorted({x.code for x in res.safety.alerts}),
                           hr=m.heart_rate_bpm, qrs=m.intervals.qrs_ms)
                for ln in ("II", "V1"):
                    if ln in s.lead_names:
                        row[ln] = dissociation_features(p_peaks(s.lead(ln), s.sampling_rate_hz), m.r_peaks_samples, s.sampling_rate_hz)
            except Exception as ex:
                row["error"] = str(ex)[:60]
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
