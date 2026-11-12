import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src"); sys.path.insert(0, sys.argv[4])
from pathlib import Path

import numpy as np
from pchain import p_chain

from cardiolens.layer1_ingestion import load_wfdb
from cardiolens.pipeline import run_pipeline

out = Path(sys.argv[1]); items = json.load(open(sys.argv[2])); k, n = map(int, sys.argv[3].split("/"))
for it in items[k::n]:
    dest = out / (it["key"] + ".json")
    if dest.exists(): continue
    row = {"key": it["key"], "labels": it["labels"], "group": it["group"]}
    try:
        sig = load_wfdb(it["path"])
        if it.get("age") is not None: sig.patient.age_years = it["age"]
        res = run_pipeline(sig, mode="A"); sg = res.signal; m = res.measurements
        row["codes"] = sorted(res.findings.codes()); row["hr"] = m.heart_rate_bpm
        row["pc"] = p_chain({ln: sg.lead(ln).astype(np.float64) for ln in ("II", "III", "aVF", "V1") if ln in sg.lead_names},
                            np.asarray(m.r_peaks_samples), sg.sampling_rate_hz, m.intervals.qrs_ms)
    except Exception as e:
        row["error"] = str(e)[:160]
    dest.write_text(json.dumps(row, default=float))
