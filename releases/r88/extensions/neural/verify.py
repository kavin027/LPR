"""Replay the ten neural-backed selections from out-of-fold and query prediction grids."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ARMS=("full","row","ecp")
GRID=np.array([1.,.5,2.,.25,0.])
TRANSFER={"cn":[.25,.25,1.],"spider":[.5,.5,.5]}


def run(here, reference=None):
    rows=[]; regs=[]; paired=[]; draws={}
    reference=here.parent if reference is None else reference
    previous=reference/"stats/inputs/transferred_draws.csv.gz"
    seedsfile=reference/"seeds/paired.csv.gz"
    calfile=reference/"cal/predictions.csv.gz"
    def read(path):return pd.read_csv(path,dtype={"id":str,"group":str})
    history=read(previous).query("base=='DVLC'").set_index(["task","mode","seed","id"],verify_integrity=True).sort_index() if previous.exists() else None
    existing_seeds=read(seedsfile).set_index(["task","seed","id"],verify_integrity=True).sort_index() if seedsfile.exists() else None
    initial=read(calfile).set_index(["task","id"],verify_integrity=True).sort_index() if calfile.exists() else None
    for task in ["cn","spider"]:
        for design,setting in [("seed",251121),("seed",251122),("seed",251123),("registry",1),("registry",2)]:
            with np.load(here/f"{task}-{design}-{setting}.npz") as file:z=dict(file)
            uid=z["reg_ids"]; ids=z["query_ids"];g=z["reg_groups"].astype(str);qg=z["query_groups"].astype(str)
            assert len(set(uid))==75 and len(set(ids))==len(ids) and set(uid).isdisjoint(ids) and set(g).isdisjoint(qg)
            for f in range(5):
                h=z["reg_fold"]==f
                assert h.any() and set(g[h]).isdisjoint(g[~h])
            count=(z["reg_pred"]==z["reg_gold"][:,None,None]).sum(0)
            chosen=count.argmax(1)
            np.testing.assert_array_equal(chosen,z["selected"])
            pred={}
            for j,arm in enumerate(ARMS):
                transfer_index=np.flatnonzero(GRID==TRANSFER[task][j])[0]
                for flavour,idx in (("selected",chosen[j]),("unit",0),("transferred",transfer_index)):
                    pred[f"{arm}_{flavour}"]=z["query_pred"][:,j,idx]
                for k,alpha in enumerate(GRID):
                    regs.append((task,design,setting,arm,alpha,int(count[j,k]),int(k==chosen[j])))
            gold=z["query_gold"]
            if history is not None and design=="registry":
                mode={"cn":"enriched","spider":"uniform"}[task]
                d=history.loc[(task,mode,setting)].loc[ids]
                for j,exp in [("group",qg),("gold",gold),("draw_unseen",z["query_unseen"]),("original_unseen",z["original_unseen"]),("full",pred["full_unit"]),("row",pred["row_unit"])]+[(f"transferred_{a}",pred[a+"_transferred"]) for a in ARMS]:
                    np.testing.assert_array_equal(d[j],exp)
            elif design=="seed" and setting!=251121 and existing_seeds is not None:
                d=existing_seeds.loc[(task,setting)].loc[ids]
                # Both negative values are archive-specific sentinel codes for an unproposable, unsupported set.
                np.testing.assert_array_equal(np.maximum(d["gold"].to_numpy(),-1),gold)
                for j,exp in [("lpr",pred["full_unit"]),("ecp",pred["ecp_unit"])]:
                    np.testing.assert_array_equal(d[j],exp)
            elif design=="seed" and setting==251121 and initial is not None:
                d=initial.loc[task].loc[ids]
                for name in ARMS:
                    for orig,local in [("fixed","unit"),("selected","selected")]:
                        np.testing.assert_array_equal(d[name+"_"+orig],pred[name+"_"+local])
            for cohort,mask in [("all",np.ones(len(ids),bool)),("draw_unseen",z["query_unseen"]),("original_unseen",z["original_unseen"])]:
                n=mask.sum();groups=sorted(set(qg[mask]),key=int if task=="cn" else str)
                index=pd.Categorical(qg[mask],categories=groups).codes
                sizes=np.bincount(index)
                key=task,tuple(groups),tuple(sizes)
                if key not in draws:
                    w=np.random.default_rng(251121).multinomial(len(groups),np.full(len(groups),1/len(groups)),2000)
                    draws[key]=(w,w@sizes)
                w,den=draws[key]
                for name,p in pred.items():
                    hit=np.count_nonzero(p[mask]==gold[mask])
                    rows.append((task,design,setting,cohort,name,n,hit,100*hit/n))
                l=pred["full_selected"][mask]==gold[mask]
                for name in ["row_selected","ecp_selected","full_unit","full_transferred"]:
                    r=pred[name][mask]==gold[mask];f,h=(l&~r).sum(),(r&~l).sum()
                    delta=l.astype(int)-r.astype(int)
                    lo,hi=np.percentile(100*(w@np.bincount(index,weights=delta).astype(int))/den,[2.5,97.5])
                    paired.append((task,design,setting,cohort,name,n,l.sum(),r.sum(),100*l.mean(),100*r.mean(),f,h,100*(f-h)/n,lo,hi))
    return {"accuracy":pd.DataFrame(rows,columns="task design setting cohort method n hits accuracy".split()),
            "grid":pd.DataFrame(regs,columns="task design setting method alpha hits selected".split()),
            "paired":pd.DataFrame(paired,columns="task design setting cohort comparator n full_hits ref_hits full_acc ref_acc fixes harms pg low high".split())}


if __name__=="__main__":
    here=Path(__file__).resolve().parent
    reference=Path(sys.argv[sys.argv.index("--reference")+1]) if "--reference" in sys.argv else None
    if reference is not None:
        assert all((reference/f).is_file() for f in ["stats/inputs/transferred_draws.csv.gz","seeds/paired.csv.gz","cal/predictions.csv.gz"])
    for key,frame in run(here,reference).items():
        path=here/(key+".csv")
        if "--write" in sys.argv:frame.to_csv(path,index=False)
        else:pd.testing.assert_frame_equal(frame,pd.read_csv(path),rtol=1e-10,atol=1e-10)
    print("PASS: group-held-out neural re-selection at three seeds and two replacement registries per task.")
