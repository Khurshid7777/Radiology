import json, sys, glob
rows=[json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith("{")]
def bad(r, s, k): v=r[f"{k}_{s}"]; return v is None or v>20
print("n", len(rows))
for k in ("qrs_ms","qt_ms","rr_ms"):
    b=sum(bad(r,"before",k) for r in rows); a=sum(bad(r,"after",k) for r in rows)
    worse=[r["case"] for r in rows if not bad(r,"before",k) and bad(r,"after",k)]
    print(k, "bad before", b, "after", a, "newly bad", len(worse), worse[:6])
pc=[r for r in rows if r["slice"]=="photo_clean"]
for k in ("rr_ms","qrs_ms","qt_ms"):
    f=lambda s: sum(r[f"{k}_{s}"] or 0 for r in pc)/max(1,len(pc))
    print("photo_clean", k, "MAE before", round(f("before"),2), "after", round(f("after"),2))
