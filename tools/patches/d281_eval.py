import json,glob,itertools,collections,sys
SP=sys.argv[1]; MODE=sys.argv[2] if len(sys.argv)>2 else "tune"
L=lambda p:[json.loads(l) for f in sorted(glob.glob(SP+p)) for l in open(f)]
F={"ventricular_tachycardia","wide_complex_tachycardia","wide_qrs_tachycardia","ventricular_fibrillation"}
GATE={"atrial_fibrillation","possible_atrial_fibrillation","atrial_flutter","poor_quality_signal","no_ecg_signal"}
codes={(r["db"],r["rec"],r["s0"]):r for r in L("/vp_w.jsonl")}
vm=L("/vm_vp_*.jsonl")
for r in vm: r.update(codes=codes[(r["db"],r["rec"],r["s0"])]["codes"],alerts=codes[(r["db"],r["rec"],r["s0"])]["alerts"])
even=lambda r:int(r["rec"])%2==0
if MODE=="tune": V=[r for r in vm if r["db"]=="vfdb" and even(r)]
else: V=[r for r in vm if r["db"]=="cudb" or (r["db"]=="vfdb" and not even(r))]
mit=L("/vm_mit_*.jsonl"); ptb=L("/vm_ptb_*.jsonl"); f2=L("/vm_f2_*.jsonl")
def soft(mts,Lk,C,K,R,S):
    u=[m for m in mts if m]
    if not u: return False
    h=sum(1 for m in u if m["leak"]<=Lk and m["conc"]>=C and m["crest"]<=K and m["rms"]>=R and m["f_dom"]>=2.5)
    return h>=1 and h/len(u)>=S
def ev(p):
    vv=[r for r in V if r["cls"] in("VT","VFL","VF")]
    miss=[r for r in vv if not set(r["alerts"])&F]
    fixed=[r for r in miss if set(r["codes"])&GATE and soft(r["mts"],*p)]
    org=[r for r in V if r["cls"]=="ORG"]; orgnew=[r for r in org if not set(r["alerts"])&F and set(r["codes"])&GATE and soft(r["mts"],*p)]
    mf=[r for r in mit if not r["vf"] and soft(r["mts"],*p)]
    pf=[r for r in ptb if not r["vf"] and soft(r["mts"],*p)]
    ff=[r for r in f2 if not r["vf"] and soft(r["mts"],*p)]
    return len(miss),len(fixed),len(org),len(orgnew),len(mf),len(pf),len(ff),[r["id"] for r in ff if "NORM" in r["labels"]]
if MODE=="tune":
    res=[]
    for p in itertools.product((0.5,0.6,0.7),(0.2,0.25,0.3),(4.5,5.5,6.5),(30,60),(0.5,0.7,1.0)):
        miss,fx,no,on,mf,pf,_,_=ev(p)
        res.append((fx,-(mf+pf),p,miss,on,no,mf,pf))
    ok=[x for x in res if x[6]<=len(mit)*0.005 and x[7]<=len(ptb)*0.005 and x[4]<=6]
    ok.sort(reverse=True)
    print("mit",len(mit),"ptb",len(ptb))
    for x in ok[:12]: print("fixed",x[0],"of",x[3],"p",x[2],"orgnew",x[4],"/",x[5],"mitF",x[6],"ptbF",x[7])
else:
    p=tuple(float(v) for v in sys.argv[3].split(","))
    print(ev(p), "f2 n",len(f2))
