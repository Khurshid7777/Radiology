import json,glob,numpy as np,re,sys,itertools
SP=sys.argv[1]; par=int(sys.argv[2]) if len(sys.argv)>2 else None
R=[json.load(open(p)) for p in glob.glob(SP+"/pc/*.json")]
R=[r for r in R if "error" not in r]
AFc={"atrial_fibrillation","possible_atrial_fibrillation"}
num=lambda r:int(re.sub(r"\D","",r["key"].split("_")[-1]) or 0)
sel=lambda r: par is None or num(r)%2==par
af=lambda r: bool(set(r["codes"])&AFc)
blk={"2AVB","MOBITZ1","AVB"}
pos=[r for r in R if r["group"]=="chapman_avb" and set(r["labels"])&blk and "1AVB" not in r["labels"] and af(r) and sel(r)]
chaf=[r for r in R if r["group"]=="chapman_af" and af(r) and sel(r)]
ptaf=[r for r in R if r["group"]=="ptb_true" and af(r)]
ctl=[r for r in R if r["group"]=="chapman_avb" and not set(r["labels"])&{"1AVB","2AVB","3AVB","AVB","MOBITZ1"} and sel(r)]
print("pos",len(pos),"chAF",len(chaf),"ptbAF",len(ptaf),"ctl",len(ctl))
g=lambda r,k: (r.get("pc") or {}).get(k)
for k in ("ratio","regular","rate_bpm"):
    q=lambda rs: np.nanpercentile([g(r,k) if g(r,k) is not None else np.nan for r in rs],[10,50,90]).round(2)
    print(k,"pos",q(pos),"chAF",q(chaf),"ptAF",q(ptaf),"ctl",q(ctl))
def hit(r,a,b): 
    pc=r.get("pc")
    return pc is not None and pc["regular"]>=a and pc["ratio"]>=b and 50<=pc["rate_bpm"]<=150
res=[]
for a,b in itertools.product((0.6,0.7,0.75,0.8,0.85,0.9),(1.1,1.2,1.3,1.5)):
    res.append((sum(hit(r,a,b) for r in chaf)+sum(hit(r,a,b) for r in ptaf),a,b,sum(hit(r,a,b) for r in pos),len(pos),sum(hit(r,a,b) for r in chaf),sum(hit(r,a,b) for r in ptaf),sum(hit(r,a,b) for r in ctl)))
for x in sorted(res,key=lambda t:(t[0],-t[3]))[:14]: print("AFlost(ch+pt)=%d a=%.2f b=%.1f | blocks %d/%d | chAF %d ptAF %d | ctl %d"%x)
