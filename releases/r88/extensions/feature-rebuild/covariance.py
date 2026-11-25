"""Choose inverse covariance by group-held-out legacy-vector log density.
Freeze rule: shape maximizes conditional Gaussian log score summed over all
registry rows. Fixed tie order full, diagonal, isotropic. Its scale coefficient
comes from the original grouped exact-hit grid for that shape. Evaluate only
after freezing shape and weight. Inputs: full historical feature snapshot and
compact numerical replay. Run: python covariance87.py HISTORY COMPACT OUTPUT .
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
def run(root,compact,dest):
    spec=importlib.util.spec_from_file_location('geom',root/'round2_mechanism/compute.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    split=pd.read_csv(root/'review5_selector/folds.csv',dtype={'id':str})
    rules=pd.read_csv(compact/'decoder/results/rules.csv').set_index(['task','rule'])
    scores=[];selected={}
    for task in ['cn','spider']:
        with np.load(root/'refit_inputs'/f'{task}.npz') as az: y,p,g,ids=[az[c] for c in ['y','p','groups','register_ids']]
        fold=split[(split.corpus==task)&(split['mode']=='original')]
        np.testing.assert_array_equal(fold.id,ids)
        for k in sorted(fold.fold.unique()):
            v=fold.fold.to_numpy()==k; tr=~v
            keep=y[tr].any(0)
            z=y[:,keep].astype(float)
            res_train=p[tr]-geo.crossfit(z[tr],p[tr],g[tr])
            err=p[v]-geo.predict(z[tr],p[tr],z[v])
            cov=LedoitWolf().fit(res_train).covariance_
            for shape,h in [('full',cov),('diagonal',np.diag(cov.diagonal())),('isotropic',np.eye(len(cov))*cov.diagonal().mean())]:
                e,u=np.linalg.eigh(h)
                e=np.maximum(e,max(1.e-12,1.e-8*e.max()))
                nll=.5*(np.log(e).sum()+((err@u)**2/e).sum(1))/len(cov)
                scores.extend((task,i,k,shape,s) for i,s in zip(ids[v],nll))
        d=pd.DataFrame(scores,columns='task id fold shape nll'.split())
        avg=d[d.task==task].groupby('shape',sort=False).nll.mean()
        selected[task]=avg.idxmin()
    # No evaluation outcome enters either selection step.
    d.to_csv(dest/'registry-density.csv',index=False)
    raw=pd.read_csv(compact/'decoder/results/predictions.csv.gz',dtype={'id':str,'group':str})
    rows=[]
    for task,shape in selected.items():
        a=raw[(raw.task==task)&(raw.rule==shape)]
        for cohort,s in [('all',a),('unseen',a[a.stratum=='unseen_combination'])]:
            rows.append((task,shape,rules.at[(task,shape),'alpha'],cohort,len(s),int(s.pred.eq(s.gold).sum()),100*s.pred.eq(s.gold).mean()))
    pd.DataFrame(rows,columns='task shape alpha cohort n correct acc'.split()).to_csv(dest/'density-summary.csv',index=False)
if __name__=='__main__':
    with threadpool_limits(1): run(*(Path(arg) for arg in sys.argv[1:]))

