"""Shared-query and stratified redraw perturbations for fresh-minus-transferred Full."""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
HERE=Path(__file__).resolve().parent
SEED=251121
REPS=2000
def run():
    entries=[]
    for task,alpha in [('cn',.25),('spider',.5)]:
        with np.load(HERE/'standalone'/f'{task}.npz') as z:
            groups,idx=np.unique(z['query_groups'],return_inverse=True)
            rng=np.random.default_rng(SEED)
            q=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),REPS)
            redraw=np.zeros((REPS,20),int)
            for mode in ('uniform','enriched'):
                where=np.flatnonzero(z['mode']==mode)
                assert len(where)==10
                redraw[:,where]=rng.multinomial(10,np.full(10,.1),REPS)
            for cohort in ('All','Draw-unseen'):
                sums=[];ns=[]
                for j in range(20):
                    left=z['query_pred'][j,:,0,z['cv_selected'][j,0]]==z['query_gold'][j]
                    right=z['query_pred'][j,:,0,[1.,.5,2.,.25,0.].index(alpha)]==z['query_gold'][j]
                    mask=np.ones(len(idx),bool) if cohort=='All' else z['query_unseen'][j]
                    sums.append(np.bincount(idx[mask],weights=left[mask].astype(int)-right[mask].astype(int),minlength=len(groups)))
                    ns.append(np.bincount(idx[mask],minlength=len(groups)))
                sums=np.array(sums);ns=np.array(ns)
                denom=q@ns.T
                assert denom.min()>0
                per=100*(q@sums.T)/denom
                point=100*np.mean(sums.sum(1)/ns.sum(1))
                for name,arr in [('fixed-panel',per.mean(1)),('redraw-and-groups',(redraw*per).sum(1)/20)]:
                    low,high=np.percentile(arr,[2.5,97.5])
                    entries.append((task,cohort,name,point,low,high))
    return pd.DataFrame(entries,columns=['task','cohort','perturbation','mean_gain','p025','p975'])
if __name__=='__main__':
    out=HERE/'perturbations.csv';df=run()
    if '--write' in sys.argv:df.to_csv(out,index=False)
    else:pd.testing.assert_frame_equal(df,pd.read_csv(out),rtol=1e-11,atol=1e-11)
    print(df.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
