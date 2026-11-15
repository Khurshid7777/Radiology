import sys,csv,json,warnings;warnings.filterwarnings("ignore");sys.path.insert(0,"src")
import numpy as np,wfdb
from cardiolens.layer1_ingestion import load_wfdb
from cardiolens.pipeline import run_pipeline
import cardiolens.layer3_measure.measure as M
rows={r["ID"]:r for r in csv.DictReader(open("data/ludb/ludb.csv"))}
out=[]
for rid in sorted(rows,key=int):
    rh=rows[rid]["Rhythms"].strip().split("\n")[0]
    a=wfdb.rdann(f"data/ludb/data/{rid}","ii"); s=a.symbol; x=a.sample; prs=[]
    for i,cc in enumerate(s):
        if cc=="p" and i>0 and s[i-1]=="(":
            for j in range(i+1,min(i+8,len(s))):
                if s[j]=="N" and s[j-1]=="(": prs.append((x[j-1]-x[i-1])/500*1000); break
    r={"id":int(rid),"rh":rh,"gold":float(np.median(prs)) if prs else None,
       "avb":"i degree av block" in rows[rid]["Conduction abnormalities"].lower().replace("iii","")}
    for tag,mb in (("a",True),("b",False)):
        M.P_MEDIAN_BEAT=mb
        res=run_pipeline(load_wfdb(f"data/ludb/data/{rid}"),mode="A"); m=res.measurements
        r[tag]=m.intervals.pr_ms; r[tag+"_codes"]=sorted(c for c in res.findings.codes() if "av_block" in c or "pr_" in c)
        if tag=="a": r["c"]=m.pr_multilead_ms
    out.append(r); print(json.dumps(r),flush=True)
