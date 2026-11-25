"""Refit the grouped profile residuals and reproduce the held-out density selector."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def predict(x, targets, query):
    xc=x-x.mean(axis=0)
    slopes=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(targets-targets.mean(axis=0)))
    return targets.mean(axis=0)+(query-x.mean(axis=0))@slopes

def shrinkage(residual):
    # Ledoit-Wolf covariance, using the population normalisation of empirical covariance.
    x=residual-residual.mean(axis=0)
    n,p=x.shape
    sample=x.T@x/n
    mu=np.trace(sample)/p
    dispersion=np.square(sample-mu*np.eye(p)).sum()/p
    if dispersion==0:
        return sample
    loss=(np.square(x).sum(axis=1)**2).sum()/n-np.square(sample).sum()
    fraction=np.clip(loss/(p*n*dispersion),0.,1.)
    return (1-fraction)*sample+fraction*mu*np.eye(p)

def verify():
    split=pd.read_csv(ROOT/'primary/cal/folds.csv',dtype={'id':str})
    rows=[]
    for task in ('cn','spider'):
        with np.load(HERE/f'{task}.npz', allow_pickle=False) as data:
            y,p,g,ids=(data[k] for k in ('y','p','groups','register_ids'))
            assignment=split[(split.corpus==task)&(split['mode']=='original')]
            np.testing.assert_array_equal(ids,assignment.id)
            for fold in sorted(assignment.fold.unique()):
                held=assignment.fold.eq(fold).to_numpy()
                xt=y[~held][:,y[~held].any(axis=0)]
                yt=p[~held]
                groups=g[~held]
                residual=np.empty_like(yt)
                for group in np.unique(groups):
                    single=groups==group
                    residual[single]=yt[single]-predict(xt[~single],yt[~single],xt[single])
                error=p[held]-predict(xt,yt,y[held][:,y[~held].any(axis=0)])
                cov=shrinkage(residual)
                for name,matrix in [('full',cov),('diagonal',np.diag(np.diag(cov))),
                                    ('isotropic',np.eye(len(cov))*np.diag(cov).mean())]:
                    eig,basis=np.linalg.eigh(matrix)
                    eig=eig.clip(min=max(1e-12,eig.max()*1e-8))
                    density=.5*(np.log(eig).sum()+(np.square(error@basis)/eig).sum(axis=1))/len(cov)
                    rows.extend((task,rid,fold,name,v) for rid,v in zip(ids[held],density))
    detailed=pd.DataFrame(rows,columns='task id fold shape nll'.split())
    reference=pd.read_csv(HERE/'registry-density.csv',dtype={'id':str})
    pd.testing.assert_frame_equal(detailed,reference,check_exact=False,atol=1e-9,rtol=1e-9)
    choices=detailed.groupby(['task','shape'],sort=False).nll.mean()
    decisions=pd.read_csv(ROOT/'primary/decoder/results/predictions.csv.gz')
    rules=pd.read_csv(ROOT/'primary/decoder/results/rules.csv').set_index(['task','rule'])
    expected=pd.read_csv(HERE/'density-summary.csv')
    for row in expected.itertuples():
        assert choices.loc[row.task].idxmin()==row.shape
        assert rules.at[(row.task,row.shape),'alpha']==row.alpha
        d=decisions[(decisions.task==row.task)&(decisions.rule==row.shape)]
        if row.cohort=='unseen':d=d[d.stratum.eq('unseen_combination')]
        assert (len(d), int(d.pred.eq(d.gold).sum()))==(row.n,row.correct)
        np.testing.assert_allclose(100*d.pred.eq(d.gold).mean(),row.acc)
    print('PASS: 450 held-out Gaussian densities, shape selection and corresponding accuracies.')
if __name__=='__main__':verify()
