"""D319: score rows of run.py — sensitivity by beat class, extra detections per split and database."""
import collections
import json
import sys

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print("rows", len(rows), "errors", sum("error" in r for r in rows))
agg = collections.defaultdict(lambda: {"N": [0, 0], "V": [0, 0], "S": [0, 0], "F": [0, 0], "det": 0, "extra": 0, "win": 0, "win_vmiss": 0})
for r in rows:
    if "got" not in r:
        continue
    for key in ((r["split"], r["db"]), (r["split"], "all")):
        a = agg[key]; a["win"] += 1; a["det"] += r["n_det"]; a["extra"] += r["extra"]
        for c in "NVSF":
            a[c][0] += r["got"][c][0]; a[c][1] += r["got"][c][1]
        if r["got"]["V"][0] > r["got"]["V"][1]:
            a["win_vmiss"] += 1
for key in sorted(agg):
    a = agg[key]
    s = " ".join(f"{c} {a[c][1]}/{a[c][0]}={a[c][1] / max(1, a[c][0]):.3f}" for c in "NVS")
    print(f"{key[0]:6s} {key[1]:9s} win {a['win']:5d} | {s} | extra {a['extra']} ({a['extra'] / max(1, a['det']):.3f}) | windows with a missed V {a['win_vmiss']}")
