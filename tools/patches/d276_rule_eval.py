"""D276 frozen rule (NOT accepted), evaluated offline on tools/p_morph/pbeat.py rows.

    python tools/patches/d276_rule_eval.py <rows_dir>
"""
import json,glob,collections
L=("II","III","aVF","I","aVR","V1")
BAD={"atrial_fibrillation","possible_atrial_fibrillation","atrial_flutter","wide_complex_tachycardia","supraventricular_tachycardia","ventricular_tachycardia","wide_qrs_tachycardia"}
def load(p):
    R=[]
    for f in sorted(glob.glob(p)): R+=[json.loads(l) for l in open(f)]
    return [r for r in R if "error" not in r]
def mx(r): return max([r[l][t] for l in L if l in r for t in("pos","neg")]+[0])
def fires(r):
    return (r["hr"] and 100<=r["hr"]<140 and (r["qrs"] or 999)<120 and r["cv"] is not None and r["cv"]<.05
            and "II" in r and mx(r)<.25 and not (set(r["codes"])&BAD))
FAM={"SVT","AVNRT","AVRT","AT","PSVT"}
def grp(r):
    S=set(r["labels"])
    if S&FAM: return "FAM"
    for k in("STACH","AFL","SR"):
        if k in S: return k
    return "o"
import sys
SP=sys.argv[1]  # dir with pbeat.py rows: pb_ch_*.jsonl, pb_f2_*.jsonl
ch=[r for r in load(SP+'/pb_ch_*.jsonl') if int(r['id'][2:])%2==1]
tot=collections.Counter(grp(r) for r in ch); f=collections.Counter(grp(r) for r in ch if fires(r))
band=sum(1 for r in ch if grp(r)=="FAM" and r["hr"] and 100<=r["hr"]<140 and (r["qrs"] or 999)<120)
print("ODD tot",dict(tot),"fires",dict(f),"FAM band",band,"gain",round(f["FAM"]/band,3),"STACH pp",round(100*f["STACH"]/tot["STACH"],1))
fr=load(SP+"/pb_f2_*.jsonl"); print("fresh rows",len(fr))
for lab in("STACH","NORM","1AVB","AFIB","SR","SARRH","PAC"):
    g=[r for r in fr if lab in r["labels"]]; k=[r["id"] for r in g if fires(r)]
    print(lab,len(g),len(k),round(100*len(k)/max(1,len(g)),1),k[:5])
