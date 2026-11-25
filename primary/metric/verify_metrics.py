"""Check the metric audit from its published fold predictions and query ledgers."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def check(p):
    grid = pd.read_csv(p/'selection.csv')
    folds = pd.read_csv(p/'oof.csv.gz', dtype={'id': str})
    q = pd.read_csv(p/'predictions.csv.gz',dtype={'id':str,'group':str})
    summary = pd.read_csv(p/'metrics.csv')
    assert len(q)==4711 and not q.duplicated(['task','id']).any()
    assert q.diagonal_fixed.equals(q.diag) and q.isotropic_fixed.equals(q.sphere) and q.full_fixed.equals(q.lpr)
    keys = ['task','method','alpha']
    rec = (folds.assign(hit=folds.prediction.eq(folds.gold)).groupby(keys,as_index=False).agg(observed=('hit','sum'),n=('hit','size')))
    grid = grid.merge(rec,on=keys,validate='one_to_one')
    assert (grid.n==75).all() and (grid.observed==grid.hits).all()
    grid['tie'] = grid.alpha.map({1:0,.5:1,2:2,.25:3,0:4})
    best = grid.sort_values(['hits','tie'],ascending=[False,True]).drop_duplicates(['task','method'])
    assert best.selected.all() and grid.selected.sum()==len(best)==6
    for row in summary.itertuples():
        z=q[q.task==row.task]
        if row.cohort=='unseen':
            z=z[z.stratum=='unseen_combination']
        groups=sorted(z.group.unique(),key=int if row.task=='cn' else str)
        ix=pd.Categorical(z.group,categories=groups).codes
        gnum=len(groups)
        reps=np.random.default_rng(251121).multinomial(gnum,np.full(gnum,1/gnum),2000)
        full=z.full_selected.eq(z.gold).astype(int); ref=z[row.metric+'_selected'].eq(z.gold).astype(int)
        delta=full-ref
        ci=np.percentile(100*(reps@np.bincount(ix,weights=delta))/(reps@np.bincount(ix)),[2.5,97.5])
        assert (len(z),ref.sum(),(delta>0).sum(),(delta<0).sum())==(row.n,row.correct,row.fixes,row.harms)
        np.testing.assert_allclose([100*ref.mean(),100*delta.mean(),*ci],[row.accuracy,row.full_minus_metric,row.lo,row.hi],atol=1.e-11)
    print('PASS: matched-covariance five-fold selections, paired counts and intervals.')
if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('results',type=Path,nargs='?',default=Path(__file__).resolve().parent/'results')
    check(args.parse_args().results)
