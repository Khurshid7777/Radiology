"""D314: beat dataset — 10-s windows of INCART / MIT-BIH, annotated beats as R peaks, features of beat_shape.

    python tools/beat_shape/dataset.py {incart_even,incart_odd,mitdb} --out FILE.npz [--shard k/n]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools/vt_bench"))
import run as vb

BEATS = set("NLRVAaJSejE")      # F (fusion), Q, / left out: ambiguous


def main():
    warnings.filterwarnings("ignore")
    import wfdb
    from scipy.signal import resample_poly
    sys.path.insert(0, str(ROOT / "tools")); from beat_shape.beat_shape import features
    ap = argparse.ArgumentParser(); ap.add_argument("which"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1"); a = ap.parse_args(); k, n = (int(v) for v in a.shard.split("/"))
    db = "incartdb" if a.which.startswith("incart") else "mitdb"
    paths = sorted(str(p)[:-4] for p in (ROOT / vb.DBS[db]).glob("*.hea"))
    if a.which == "incart_even":
        paths = [p for p in paths if int(Path(p).name[1:]) % 2 == 0]
    elif a.which == "incart_odd":
        paths = [p for p in paths if int(Path(p).name[1:]) % 2 == 1]
    X, Y, G = [], [], []
    for path in paths[k::n]:
        rec = wfdb.rdrecord(path); ann = wfdb.rdann(path, "atr")
        fs = float(rec.fs); names = vb.lead_names(rec.sig_name)
        sig = resample_poly(rec.p_signal.T * 1000.0, 500, int(fs), axis=1)
        samp = np.round(np.asarray(ann.sample) * 500.0 / fs).astype(int)
        sym = np.asarray(ann.symbol)
        keep = np.isin(sym, list(BEATS))
        samp, sym = samp[keep], sym[keep]
        w = 5000
        for s0 in range(0, sig.shape[1] - w, w):
            m = (samp >= s0) & (samp < s0 + w)
            if m.sum() < 5:
                continue
            leads = {nm: sig[i, s0:s0 + w] for i, nm in enumerate(names)}
            f, ok = features(leads, samp[m] - s0, 500.0)
            if len(ok):
                X.append(f); Y.append((sym[m][ok] == "V").astype(np.int8)); G += [Path(path).name] * len(ok)
    np.savez_compressed(a.out, X=np.vstack(X), y=np.concatenate(Y), g=np.asarray(G))


if __name__ == "__main__":
    main()
