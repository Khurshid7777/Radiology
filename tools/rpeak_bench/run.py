"""D319: R-peak detection of the site's measure() on 10-s windows of MIT-BIH / INCART / SVDB against beat annotations.

    python tools/rpeak_bench/run.py --out rows.jsonl [--shard k/n] [--step 60]
Rows (numbers only): db, record, window start, labelled beats by class, matched by class, extra detections.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools/vt_bench"))
import run as vb

CLS = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "V": "V", "E": "V", "A": "S", "a": "S", "J": "S", "S": "S", "F": "F"}
TOL_S = 0.075


def main():
    warnings.filterwarnings("ignore")
    import wfdb

    from cardiolens.layer3_measure import measure as measure_fn
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--step", type=float, default=60.0)
    a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    paths = sorted((db, str(p)[:-4]) for db in ("mitdb", "incartdb", "svdb") for p in (ROOT / vb.DBS[db]).glob("*.hea"))
    with a.out.open("w") as fh:
        for db, path in paths[k::n]:
            rec = Path(path).name
            ann = wfdb.rdann(path, "atr"); hdr = wfdb.rdheader(path); fs = float(hdr.fs)
            w = int(vb.WIN_S * fs)
            samp = np.asarray(ann.sample); sym = np.asarray(ann.symbol)
            for s0 in range(int(fs), int(hdr.sig_len) - w, int(a.step * fs)):
                m = (samp >= s0) & (samp < s0 + w) & np.isin(sym, list(CLS))
                if m.sum() < 3:
                    continue
                row = {"db": db, "rec": rec, "s0": s0, "split": vb.split({"db": db, "rec": rec})}
                try:
                    sig = vb.load(path, s0, w)
                    mm = measure_fn(sig)
                    rp = np.asarray(mm.r_peaks_samples or [], float) * fs / sig.sampling_rate_hz
                    lab = samp[m] - s0; cls = [CLS[x] for x in sym[m]]
                    tol = TOL_S * fs
                    edge = (lab >= 0.15 * fs) & (lab <= w - 0.15 * fs)      # beats at the window edge are not judged
                    got = {c: [0, 0] for c in "NVSF"}
                    used = set(); miss = []
                    for i, (x, c, e) in enumerate(zip(lab, cls, edge, strict=True)):
                        if not e:
                            continue
                        got[c][0] += 1
                        j = int(np.argmin(np.abs(rp - x))) if rp.size else None
                        if j is not None and abs(rp[j] - x) <= tol:
                            got[c][1] += 1; used.add(j)
                        else:
                            prev_rr = (x - lab[i - 1]) / fs * 1000 if i else None
                            miss.append([c, None if j is None else round(abs(rp[j] - x) / fs * 1000), round(x / fs, 2),
                                         None if prev_rr is None else round(prev_rr)])
                    inner = [j for j, r in enumerate(rp) if 0.15 * fs <= r <= w - 0.15 * fs]
                    extra = [j for j in inner if j not in used and (lab.size == 0 or np.min(np.abs(lab - rp[j])) > tol)]
                    row.update(got=got, n_det=len(inner), extra=len(extra), miss=miss, n_lab=int(edge.sum()))
                except Exception as ex:
                    row["error"] = f"{type(ex).__name__}: {str(ex)[:60]}"
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
