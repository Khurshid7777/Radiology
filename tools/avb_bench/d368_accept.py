"""D368 acceptance: the frozen rule (D367 tuning) on odd Chapman, odd af_bench and the PTB-XL rhythm bench.

    python tools/avb_bench/d368_accept.py <dump dir>     # dumps from resid_feats.py: f_avb_*, f_afodd_*, f_ptb_*
"""
import glob
import json
import sys

import numpy as np

FS = 500
SKIP = {"atrial_fibrillation", "atrial_flutter", "paced_rhythm", "possible_atrial_fibrillation"}


def load(d, w):
    return [json.loads(line) for f in sorted(glob.glob(f"{d}/f_{w}_*.jsonl")) for line in open(f) if "error" not in line]


def num(i):
    return int("".join(c for c in str(i) if c.isdigit()))


def feats(r):
    rp = np.array(r.get("rp") or [], int)
    best = None
    for ln in ("II", "V1", "III", "aVF"):
        if ln in r:
            p = np.array(r[ln]["grid"], int)
            if best is None or len(p) > len(best):
                best = p
    if best is None or len(best) < 4 or len(rp) < 3:
        return None
    p = best
    ratio = float(np.median(np.diff(rp)) / np.median(np.diff(p)))
    prs = [(q - p[(p < q - 0.08 * FS) & (p > q - 0.45 * FS)].max()) / FS
           for q in rp if ((p < q - 0.08 * FS) & (p > q - 0.45 * FS)).any()]
    blocked = sum(1 for t in p if t < rp.max() and (not (rp > t).any() or (rp[rp > t][0] - t) / FS > 0.45))
    return dict(ratio=ratio, pr_share=len(prs) / len(rp), blocked=blocked,
                pr_iqr=float(np.subtract(*np.percentile(prs, [75, 25]))) if len(prs) >= 3 else None)


def rule(r):
    codes = set(r["codes"])
    if codes & SKIP:
        if "paced_rhythm" in codes or "atrial_flutter" in codes or "possible_atrial_fibrillation" in codes:
            return None
        if not (r.get("hr") and r["hr"] < 50):
            return None
    f = feats(r)
    if not f or f["blocked"] < 2 or f["ratio"] < 1.1:
        return None
    return "complete" if (f["pr_iqr"] is None or f["pr_iqr"] > 0.08 or f["pr_share"] < 0.5) else "second"


def main():
    d = sys.argv[1]
    out = {}
    avb = [r for r in load(d, "avb") if num(r["id"]) % 2 == 1]

    def cls(lab):
        s = set(lab)
        return "3AVB" if "3AVB" in s else ("2AVB" if s & {"2AVB", "2AVB1"} else "ctrl")
    g3 = [r for r in avb if cls(r["labels"]) == "3AVB"]
    g2 = [r for r in avb if cls(r["labels"]) == "2AVB"]
    gc = [r for r in avb if cls(r["labels"]) == "ctrl"]
    core3 = sum("complete_av_block" in r["codes"] for r in g3) / len(g3)
    new3 = sum(rule(r) == "complete" or "complete_av_block" in r["codes"] for r in g3) / len(g3)
    core2 = sum(bool({"complete_av_block", "probable_av_block_2to1"} & set(r["codes"])) for r in g2) / len(g2)
    new2 = sum(rule(r) is not None or bool({"complete_av_block", "probable_av_block_2to1"} & set(r["codes"])) for r in g2) / len(g2)
    out["chapman_3avb"] = {"n": len(g3), "core": round(core3, 3), "new": round(new3, 3), "gain": round(new3 - core3, 3)}
    out["chapman_2avb"] = {"n": len(g2), "core": round(core2, 3), "new": round(new2, 3)}
    out["chapman_sb_sr"] = {"n": len(gc), "false_complete_pct": round(100 * sum(rule(r) == "complete" for r in gc) / len(gc), 2),
                            "false_second_pct": round(100 * sum(rule(r) == "second" for r in gc) / len(gc), 2)}
    af = load(d, "afodd")
    out["af_bench_odd"] = {lab: {"n": len(g), "block_pct": round(100 * sum(rule(r) is not None for r in g) / max(1, len(g)), 2)}
                           for lab in ("AFIB", "AFL", "PAC", "PVC", "SARRH") for g in [[r for r in af if r["labels"] == [lab]]]}
    ptb = load(d, "ptb")
    res = {}
    for lab in ("SBRAD", "NORM", "1AVB", "AFIB", "2AVB", "3AVB"):
        g = [r for r in ptb if lab in r["labels"]]
        res[lab] = {"n": len(g), "block_pct": round(100 * sum(rule(r) is not None for r in g) / max(1, len(g)), 2),
                    "complete_pct": round(100 * sum(rule(r) == "complete" for r in g) / max(1, len(g)), 2)}
    norm = [r for r in ptb if "NORM" in r["labels"]]
    before = sum(bool(r["alerts"]) for r in norm)
    after = sum(bool(r["alerts"]) or rule(r) == "complete" for r in norm)
    res["NORM_alert_pp"] = round(100 * (after - before) / max(1, len(norm)), 2)
    out["ptbxl"] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
