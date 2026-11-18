"""D304: RR group-beating check over the rows of rr_dump.py (run it in the folder holding avb/ptb/fresh/fresh2.jsonl)."""
import json

import numpy as np


def load(f): return [json.loads(l) for l in open(f) if '"error"' not in l]
rows = {w: load(f"{w}.jsonl") for w in ("avb","ptb","fresh","fresh2")}
def grouped(rr, long_k=1.3, cv_max=0.06, spread=0, min_long=3, ratio_hi=2.1):
    rr = np.asarray(rr, float)
    if rr.size < 7: return False
    base = np.median(rr[rr <= np.percentile(rr, 60)])
    lg = rr > long_k * base
    idx = np.flatnonzero(lg)
    if idx.size < min_long or idx.size > rr.size / 2: return False
    if (np.diff(idx) == 1).any(): return False          # two long in a row: not group beating
    short = rr[~lg]
    if short.std() / short.mean() > cv_max: return False
    if not (long_k <= rr[lg].mean() / short.mean() <= ratio_hi): return False
    if rr[lg].std() / rr[lg].mean() > cv_max * 1.5: return False
    gaps = np.diff(idx)                                  # beats per group
    return gaps.size == 0 or gaps.max() - gaps.min() <= spread
af = lambda r: "atrial_fibrillation" in r["codes"]
avb = rows["avb"]; names = sorted(r["id"] for r in avb); half = {n: i % 2 for i, n in enumerate(names)}
blk = lambda r: bool(set(r["labels"]) & {"2AVB", "2AVB1"})
def ev(par, **kw):
    b = [r for r in avb if blk(r) and half[r["id"]] == par and af(r)]
    p = [r for r in rows["ptb"] if "AFIB" in r["labels"] and int(r["id"]) % 2 == par and af(r)]
    return sum(grouped(r["rr"], **kw) for r in b), len(b), sum(grouped(r["rr"], **kw) for r in p), len(p)
for kw in [dict(), dict(spread=1), dict(cv_max=0.08), dict(cv_max=0.08, spread=1), dict(min_long=2), dict(min_long=2, cv_max=0.08, spread=1),
           dict(cv_max=0.10, spread=1), dict(long_k=1.25, cv_max=0.08, spread=1)]:
    print(kw, ev(0, **kw))
print("---")
for kw in [dict(spread=99), dict(spread=99, cv_max=0.08), dict(spread=99, min_long=2), dict(spread=99, cv_max=0.08, min_long=2),
           dict(spread=99, cv_max=0.10, min_long=2), dict(spread=99, cv_max=0.08, min_long=2, long_k=1.25)]:
    print(kw, ev(0, **kw))
