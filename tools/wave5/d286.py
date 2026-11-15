"""D286: blocked PAC behind a "sinus pause" — MIT-BIH windows with a non-conducted P ("x") and windows with a true
sinus pause >= 2 s (sinus rhythm by annotation, no "x"); per window: pipeline codes and the blocked-PAC check."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import numpy as np, wfdb
from vt_bench.run import load, rhythm_spans
from cardiolens.pipeline import run_pipeline
from cardiolens.arrhythmia.descriptive import blocked_pac_in_pause
BEATS = set("NLRVAaJSejE/fFQ")
out = open(sys.argv[1], "w")
dbs = {"mitdb": "data/mitdb", "svdb": "data/svdb", "incartdb": "data/incartdb"}
for db, d in dbs.items():
    for hea in sorted((ROOT / d).glob("*.hea")):
        p = str(hea)[:-4]; a = wfdb.rdann(p, "atr"); hdr = wfdb.rdheader(p); w = int(10 * hdr.fs)
        spans = rhythm_spans(a, hdr.sig_len); per = {"x": 0, "pause": 0}
        beats = [(s, y) for s, y in zip(a.sample, a.symbol) if y in BEATS]
        for s0 in range(0, hdr.sig_len - w, w):
            sym = [y for s, y in zip(a.sample, a.symbol) if s0 <= s < s0 + w]
            rh = next((r for x0, x1, r in spans if x0 <= s0 < x1), "?")
            if rh.startswith("AF"):
                continue
            bs = [s for s, y in beats if s0 <= s < s0 + w]
            long_rr = max(np.diff(bs)) / hdr.fs if len(bs) > 2 else 0
            kind = "x" if "x" in sym else ("pause" if long_rr >= 2.0 and rh == "N" else None)
            if kind is None or per[kind] >= (40 if kind == "x" else 6):
                continue
            per[kind] += 1
            res = run_pipeline(load(p, s0, w), mode="A")
            codes = res.findings.codes()
            bp = blocked_pac_in_pause(res.signal, res.measurements) if "sinus_pause" in codes else None
            out.write(json.dumps({"db": db, "rec": hea.stem, "s0": s0, "kind": kind, "pause_named": "sinus_pause" in codes,
                                  "blocked": bp, "alerts": sorted({x.code for x in res.safety.alerts})}) + "\n")
            out.flush()
