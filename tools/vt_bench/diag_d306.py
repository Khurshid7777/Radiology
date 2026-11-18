"""D306: where a labelled V run is lost on the way to the run detector (tuning part of D264 only).

    python tools/vt_bench/diag_d306.py --out rows.jsonl [--shard k/n]
Rows: record, window, labelled run, how many V beats the core's R-peaks caught, their template correlation, whether the
detector found a run, codes. Numbers only.
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools/vt_bench"))
import run as vb  # noqa: E402


def corr_profile(sig, rp, fs):
    """The run detector's per-beat template correlation (same template rule as detect_ventricular_runs)."""
    from cardiolens.arrhythmia import beats as B
    rp = np.asarray(rp, int)
    leads = [n for n in B._RUN_LEADS if n in sig.lead_names] or list(sig.lead_names[:3])
    half = int(B._PVC_HALF_S * fs); ns = sig.signal.shape[1]
    ok = [i for i, r in enumerate(rp) if r - half >= 0 and r + half < ns]
    corr = np.full(len(rp), np.nan)
    if len(ok) < 5:
        return corr
    segs = {n: np.stack([sig.lead(n)[rp[i] - half:rp[i] + half].astype(float) for i in ok]) for n in leads}
    rr = np.diff(rp).astype(float); med = float(np.percentile(rr, B._RUN_TEMPLATE_RR_PCT)) or 1.0
    tn = [k for k, i in enumerate(ok) if i >= 1 and rr[i - 1] >= B._RUN_TEMPLATE_PRE_MIN * med]
    pick = tn if len(tn) >= 3 else list(range(len(ok)))
    tm = {n: np.median(segs[n][pick], axis=0) for n in leads}
    for k, i in enumerate(ok):
        corr[i] = np.mean([np.corrcoef(segs[n][k], tm[n])[0, 1] for n in leads])
    return corr


def main():
    warnings.filterwarnings("ignore")
    import wfdb
    from cardiolens.pipeline import run_pipeline
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    paths = sorted((db, str(p)[:-4]) for db, d in vb.DBS.items() for p in (ROOT / d).glob("*.hea")
                   if db in ("mitdb", "incartdb"))
    with a.out.open("w") as fh:
        for db, path in paths[k::n]:
            rec = Path(path).name
            if db == "incartdb" and int(rec[1:]) % 2:
                continue
            wins, fs = vb.windows(path)
            ann = wfdb.rdann(path, "atr")
            for kind, s0, tr in wins:
                if kind != "vt_run":
                    continue
                w = int(vb.WIN_S * fs)
                v = [int(s) - s0 for s, y in zip(ann.sample, ann.symbol, strict=True) if y == "V" and s0 <= s < s0 + w]
                row = {"db": db, "rec": rec, "s0": s0, **tr}
                try:
                    res = run_pipeline(vb.load(path, s0, w), mode="A"); m = res.measurements
                    sig = res.signal; f2 = sig.sampling_rate_hz
                    rp = np.asarray(m.r_peaks_samples or [], int)
                    vt = np.asarray(v, float) * f2 / fs
                    hit = [int(np.argmin(np.abs(rp - x))) for x in vt if rp.size and np.min(np.abs(rp - x)) <= 0.06 * f2]
                    corr = corr_profile(sig, rp, f2)
                    row.update(codes=sorted(res.findings.codes()), n_v=len(v), v_caught=len(set(hit)),
                               v_corr=[None if np.isnan(corr[i]) else round(float(corr[i]), 3) for i in sorted(set(hit))],
                               n_corr_low=int(np.nansum(corr <= 0.80)), n_r=int(rp.size), n_lab=tr.get("n"),
                               runs=[list(x) for x in __import__("cardiolens.arrhythmia.beats", fromlist=["x"])
                                     .detect_ventricular_runs(sig, rp, f2, recovered=m.r_peaks_recovered)])
                except Exception as ex:
                    row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
