"""Check held-out choices, continuous objectives and paired evaluation accuracies."""
from pathlib import Path
import csv,sys
import numpy as np
import pandas as pd

def verify(root):
    cv=pd.read_csv(root/'cv.csv')
    regs=pd.read_csv(root/'oof.csv.gz',dtype={'id':str})
    assert len(regs)==sum(75 for _ in cv.itertuples())
    folds=(regs.assign(hit=regs.pred==regs.gold).groupby(['task','arm','alpha'])
           .agg(hits_from_masks=('hit','sum'),n=('id','nunique'),eligible=('reachable','sum')).reset_index())
    comp=cv.merge(folds,on=['task','arm','alpha'],validate='one_to_one')
    assert ((comp.n==75)&(comp.hits_from_masks==comp.hits)&(comp.eligible==comp.reachable)).all()
    selections=cv[cv.selected].set_index(['task','arm'])
    order={1.:0,.5:1,2.:2,.25:3,0.:4}
    win=cv.assign(priority=cv.alpha.map(order)).sort_values(['hits','priority'],ascending=[False,True]).drop_duplicates(['task','arm'])
    assert win.selected.all()
    raw=pd.read_csv(root/'loss_inputs.csv.gz',dtype={'id':str})
    for (tag,arm),d in raw.groupby(['task','arm']):
        row=selections.loc[(tag,arm)]
        grad=loss=0.; n=0
        for rid,r in d.groupby('id'):
            good=np.flatnonzero(r.candidate==r.gold)
            if not len(good): continue
            k=int(good[0])
            inc=r.inverse.to_numpy()
            score=r.pre.to_numpy()+row.continuous_alpha*inc
            shift=score-score.max(); z=np.exp(shift)
            grad+=float(z@inc/z.sum()-inc[k]);loss+=np.log(z.sum())-shift[k];n+=1
        assert n==row.reachable
        np.testing.assert_allclose(loss/n,row.mean_condloss,atol=1e-9)
        if row.continuous_alpha==0: assert grad>=-1e-7
        elif row.continuous_alpha==2: assert grad<=1e-7
        else: assert abs(grad)<1e-6
    rules=pd.read_csv(root/'rules.csv')
    metrics=['isotropic','diagonal','full']
    for tag,sub in rules.groupby('task'):
        grid=selections.loc[tag]
        for x in sub.itertuples():
            if x.rule in grid.index: assert x.alpha==grid.loc[x.rule].alpha and x.arm==x.rule
            if x.rule=='joint_hits': assert x.arm==max(metrics,key=lambda m:grid.loc[m].hits)
            if x.rule=='joint_logloss': assert x.arm==min(metrics,key=lambda m:grid.loc[m].mean_condloss)
            if x.rule in ('full_logloss','joint_logloss'): assert abs(x.alpha-grid.loc[x.arm].continuous_alpha)<1e-12
    data=pd.read_csv(root/'predictions.csv.gz',dtype={'id':str})
    met=pd.read_csv(root/'metrics.csv')
    for r in met.itertuples():
        q=data[(data.task==r.task)&(data.rule==r.rule)]
        if r.cohort!='all': q=q[q.stratum==('seen' if r.cohort=='seen' else 'unseen_combination')]
        assert (len(q),q.id.nunique(),int((q.gold==q.pred).sum()))==(r.n,r.n,r.hits)
        np.testing.assert_allclose(100*r.hits/r.n,r.accuracy,atol=1e-9)
    print('PASS: decoder held-out grid decisions and convex loss gradients, metric selections and frozen evaluations.')
if __name__=='__main__':
    verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'results')
