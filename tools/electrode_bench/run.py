"""D214: limb-electrode swaps simulated on PTB-XL records — exact sensitivity and specificity.

    python tools/electrode_bench/run.py --data data/ptbxl [--n 600] [--out reports/electrode_bench/x.json]

The limb leads are rebuilt from the electrode potentials with two electrodes exchanged (the chest leads
are kept: the Wilson terminal is the mean of RA, LA, LL and does not change when two of them swap; with
the neutral (RL) electrode the terminal changes a little, which is ignored). Each record is checked as
recorded (specificity) and with every swap (sensitivity), by ``detect_limb_swap`` alone (no pipeline).
Licensed data stay in data/; only the summary goes to reports/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")


def swap(sig, kind: str):
    """A copy of ``sig`` with the limb leads as recorded with ``kind`` electrodes exchanged."""
    s = sig.model_copy(deep=True)
    x = {ld: sig.lead(ld).astype(np.float64) for ld in ("I", "II")}
    # potentials relative to RA (RA = 0): LA = I, LL = II; RL is the reference ground (~ Wilson-free 0 offset)
    ra, la, ll = np.zeros_like(x["I"]), x["I"], x["II"]
    rl = ll                            # the right leg sits at about the left leg's potential
    e = {"RA": ra, "LA": la, "LL": ll, "RL": rl}
    a, b = {"la_ra": ("LA", "RA"), "ra_ll": ("RA", "LL"), "la_ll": ("LA", "LL"),
            "ra_rl": ("RA", "RL"), "la_rl": ("LA", "RL")}[kind]
    e[a], e[b] = e[b], e[a]
    ra, la, ll = e["RA"], e["LA"], e["LL"]
    new = {"I": la - ra, "II": ll - ra, "III": ll - la,
           "aVR": ra - (la + ll) / 2, "aVL": la - (ra + ll) / 2, "aVF": ll - (ra + la) / 2}
    arr = np.array(s.signal, dtype=np.float64)
    for ld, v in new.items():
        arr[s.lead_names.index(ld)] = v
    s.signal = arr.astype(sig.signal.dtype) if hasattr(sig.signal, "dtype") else arr
    return s


KINDS = ("la_ra", "ra_ll", "la_ll", "ra_rl", "la_rl")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer2_preprocess.quality import detect_limb_swap

    man = json.loads((a.data / "manifest.json").read_text())["records"]
    random.Random(a.seed).shuffle(man)
    man = man[: a.n]
    res = {"recorded": {}, **{k: {} for k in KINDS}}
    by_label_fp: dict[str, int] = {}
    n = 0
    for rec in man:
        try:
            sig = load_wfdb(str(a.data / rec["filename_hr"]))
        except Exception as e:  # an unreadable record is skipped, and said so
            print("skip", rec["ecg_id"], type(e).__name__, file=sys.stderr)
            continue
        n += 1
        got = detect_limb_swap(sig)
        res["recorded"][got or "none"] = res["recorded"].get(got or "none", 0) + 1
        if got:
            for lab in rec["labels"]:
                by_label_fp[lab] = by_label_fp.get(lab, 0) + 1
        for k in KINDS:
            g = detect_limb_swap(swap(sig, k))
            res[k][g or "none"] = res[k].get(g or "none", 0) + 1
    out = {"records": n, "results": res,
           "sensitivity": {k: round(res[k].get(k, 0) / n, 3) for k in KINDS},
           "any_flag_on_swapped": {k: round(1 - res[k].get("none", 0) / n, 3) for k in KINDS},
           "false_on_recorded": round(1 - res["recorded"].get("none", 0) / n, 4),
           "false_by_label": dict(sorted(by_label_fp.items(), key=lambda x: -x[1])[:12])}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
