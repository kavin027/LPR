"""Verify fivefold weight reselection from registration and query mask grids."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("full", "row", "ecp")
GRID = np.array([1., .5, 2., .25, 0.])
FIXED = {"cn":[.25,.25,1.], "spider":[.5,.5,.5]}


def replay(here, anchor=None):
    anchor = (here.parent/"stats/inputs/transferred_draws.csv.gz") if anchor is None else anchor
    history=pd.read_csv(anchor,dtype={"id":str,"group":str}).query("base=='standalone'").set_index(["task","mode","seed","id"],verify_integrity=True).sort_index() if anchor.exists() else None
    audit=pd.read_csv(here/"fold_audit.csv").set_index(["task","mode","seed","fold"],verify_integrity=True)
    draws=[]; weights=[]
    for task in FIXED:
        with np.load(here/f"{task}.npz",allow_pickle=False) as file:
            z=dict(file)
        ids=z["query_ids"]; groups=z["query_groups"]
        assert len(ids)==len(set(ids))=={"cn":2744,"spider":1967}[task]
        for i,(mode,seed) in enumerate(zip(z["mode"],z["seed"])):
            uid=z["cv_ids"][i]; g=z["cv_groups"][i]; fold=z["cv_fold"][i]
            assert len(set(uid))==75 and set(uid).isdisjoint(ids) and set(g).isdisjoint(groups)
            assert set(fold)==set(range(5))
            for j in range(5):
                held=fold==j
                assert set(g[held]).isdisjoint(g[~held])
                r=audit.loc[(task,mode,seed,j)]
                assert r.n_train==sum(~held) and r.n_held==sum(held) and r.groups==len(set(g[held]))
            hits=(z["cv_oof"][i]==z["cv_gold"][i][:,None,None]).sum(0)
            chosen=hits.argmax(1)
            np.testing.assert_array_equal(chosen,z["cv_selected"][i])
            predictions=z["query_pred"][i]; gold=z["query_gold"][i]
            named={}
            for j,arm in enumerate(ARMS):
                trans=np.where(GRID==FIXED[task][j])[0][0]
                for label,slot in [("selected",chosen[j]),("unit",0),("transferred",trans)]:
                    named[f"{arm}_{label}"]=predictions[:,j,slot]
                for slot,w in enumerate(GRID):
                    weights.append((task,mode,seed,arm,w,hits[j,slot],slot==chosen[j]))
            if history is not None:
                d=history.loc[(task,mode,seed)].loc[ids]
                for key,values in [("group",groups),("gold",gold),("original_unseen",z["query_fixed"]),("draw_unseen",z["query_unseen"][i]),("full",named["full_unit"]),("row",named["row_unit"])] + [(f"transferred_{a}",named[a+"_transferred"]) for a in ARMS]:
                    np.testing.assert_array_equal(d[key],values)
            for cohort,sel in [("all",np.ones(len(ids),bool)),("draw_unseen",z["query_unseen"][i]),("original_unseen",z["query_fixed"])]:
                for principal in ["full_selected","full_transferred","full_unit"]:
                    p=named[principal][sel]==gold[sel]
                    for ref,v in named.items():
                        r=v[sel]==gold[sel]
                        f,h=(p&~r).sum(),(r&~p).sum()
                        draws.append((task,mode,seed,cohort,principal,ref,len(p),int(p.sum()),int(r.sum()),100*p.mean(),100*r.mean(),f,h,100*(f-h)/len(p)))
    detail=pd.DataFrame(draws,columns="task mode seed cohort decoder comparator n hits ref_hits acc ref_acc fixes harms gain".split())
    grid=pd.DataFrame(weights,columns="task mode seed arm alpha hits selected".split())
    b=pd.concat([detail,detail.assign(mode="both")]).groupby(["task","mode","cohort","decoder","comparator"]).agg(
        draws=("gain","size"),n_min=("n","min"),n_max=("n","max"),acc=("acc","mean"),ref_acc=("ref_acc","mean"),fixes=("fixes","mean"),harms=("harms","mean"),gain=("gain","mean"),sd=("gain","std"),lo=("gain","min"),hi=("gain","max"),wins=("gain",lambda a:(a>0).sum()),ties=("gain",lambda a:(a==0).sum()),losses=("gain",lambda a:(a<0).sum())).reset_index()
    return {"selected_grid":grid,"paired_draws":detail,"summary":b}

if __name__=="__main__":
    here=Path(__file__).resolve().parent
    anchor=Path(sys.argv[sys.argv.index("--anchor")+1]) if "--anchor" in sys.argv else None
    for key,frame in replay(here,anchor).items():
        path=here/(key+".csv.gz")
        if "--write" in sys.argv:
            frame.to_csv(path,index=False,compression={"method":"gzip","mtime":0})
        else:
            pd.testing.assert_frame_equal(frame,pd.read_csv(path),rtol=1e-10,atol=1e-10)
    print("PASS: 40 grouped registries, 200 disjoint folds, 600 registered weight cells and 94,220 query predictions per configuration.")

