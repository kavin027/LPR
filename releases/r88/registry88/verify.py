"""Replay fixed r87 registration selections and paired contrasts; accepts --write."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRID = np.array([1., .5, 2., .25, 0.])
FIXED = {"cn": .25, "spider": .5}

def folds(ids, groups, cvfold, qids, qgroups):
    assert len(set(ids)) == len(ids) == 75 and set(ids).isdisjoint(qids)
    assert set(groups).isdisjoint(qgroups) and set(cvfold)==set(range(5))
    for f in range(5):
        assert set(groups[cvfold==f]).isdisjoint(groups[cvfold!=f])

def contrast(kind, task, mode, seed, reg, rgold, selected, pred, gold, unseen):
    best=(reg==rgold[:,None,None]).sum(axis=0).argmax(axis=1)
    np.testing.assert_array_equal(best,selected)
    for j in (1,2):
        np.testing.assert_array_equal(pred[:,0,4],pred[:,j,4])
    choices=[pred[:,j,best[j]] for j in range(3)]
    ref={"ECP":choices[2],"Row":choices[1],"Forward":pred[:,0,4],
         "Transferred":pred[:,0,np.flatnonzero(GRID==FIXED[task])[0]]}
    for c,mask in (("All",np.ones(len(gold),dtype=bool)),("Draw-unseen",unseen)):
        for name,yp in ref.items():
            left=choices[0][mask]==gold[mask];right=yp[mask]==gold[mask]
            fixes=int((left&~right).sum());harms=int((right&~left).sum())
            yield (kind,task,mode,int(seed),c,name,*GRID[best],len(left),int(left.sum()),
                   int(right.sum()),fixes,harms,100*(fixes-harms)/len(left))
def run():
    rows=[]
    for task in FIXED:
        with np.load(ROOT/'standalone'/f'{task}.npz') as z:
            for i,(mode,seed) in enumerate(zip(z['mode'],z['seed'])):
                folds(z['cv_ids'][i],z['cv_groups'][i],z['cv_fold'][i],z['query_ids'],z['query_groups'])
                rows.extend(contrast("standalone",task,mode,seed,z['cv_oof'][i],z['cv_gold'][i],
                           z['cv_selected'][i],z['query_pred'][i],z['query_gold'][i],z['query_unseen'][i]))
        for file in sorted((ROOT/'neural').glob(f'{task}-*.npz')):
            _,mode,seed=file.stem.split('-')
            with np.load(file) as z:
                folds(z['reg_ids'],z['reg_groups'],z['reg_fold'],z['query_ids'],z['query_groups'])
                rows.extend(contrast('DVLC',task,mode,seed,z['reg_pred'],z['reg_gold'],z['selected'],
                                     z['query_pred'],z['query_gold'],z['query_unseen']))
    detail=pd.DataFrame(rows,columns='kind task design seed cohort reference full_weight row_weight ecp_weight n full_hits ref_hits fixes harms gain'.split())
    assert len(detail)==400
    standalone=detail.query("kind=='standalone'").assign(design='pooled')
    groups=pd.concat([detail,standalone]).groupby(['kind','task','design','cohort','reference'])
    summary=groups.gain.agg(n='count',mean='mean',sd='std',worst='min',wins=lambda x:(x>0).sum(),
                            ties=lambda x:(x==0).sum(),losses=lambda x:(x<0).sum()).reset_index()
    return detail,summary

if __name__=='__main__':
    for name,df in zip(['per_fit','summary'],run()):
        path=ROOT/(name+'.csv')
        if '--write' in sys.argv: df.to_csv(path,index=False)
        else:pd.testing.assert_frame_equal(df,pd.read_csv(path),rtol=1e-11,atol=1e-11)
    print("PASS: 40 standalone draws, 10 neural fits, 400 paired contrasts.")
