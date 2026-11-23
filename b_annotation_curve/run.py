"""Fit the three service variants on exactly the same frozen B registration paths."""
import gzip
import importlib.util
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from freeze import OUT, BASE, read_np


def indicators(records, field, labels):
    return np.asarray([[s in r[field] for s in labels] for r in records], float)


def classify(api, x, t, p, pq, old, y, groups):
    # Leave-whole-group-out ridge on B only, same centered ridge as historical_readout.
    held, raw = api.calibration(x, y, groups, t)
    raw = raw.clip(0, 1)
    cv = {k:np.all((held.clip(0,1)>=k/10)==y,axis=1).sum() for k in range(1,10)}
    k = max(cv,key=lambda k:(cv[k], -abs(k-5), k))
    _, _, prob = api.forward(x,t,p,pq,old,y,groups)
    bits = 1 << np.arange(y.shape[1])
    yc = y-y.mean(0)
    b = solve(yc.T @ yc + np.eye(y.shape[1]), yc.T @ (p-p.mean(0)), assume_a='pos')
    loo, _ = api.calibration(y,p,groups,y[:0])
    cov = LedoitWolf().fit(p-loo).covariance_
    e, v = np.linalg.eigh(cov)
    white = v / np.sqrt(np.maximum(e, max(1e-12, 1e-8*e.max())))
    w = b @ white
    obs = (pq-p.mean(0)+y.mean(0) @ b) @ white
    pred = []
    for pp, ob, opts in zip(prob,obs,api.options(prob,y),strict=True):
        membership = ((opts[:,None] & bits)!=0).astype(float)
        val = membership @ np.log(pp/(1-pp)) - .5*((membership @ w-ob)**2).sum(1)
        pred.append(opts[val.argmax()])
    return dict(direct=(raw>=.5) @ bits, direct_cv=(raw>=k/10) @ bits,forward=(prob>=.5) @ bits,inverse=pred), k/10


def run(z, api):
    protocol=json.loads((OUT/'freeze.json').read_bytes())
    snapshots=[]; rows=[]; changes=[]; intervals=[]; vocabs=[]
    for ds,spec in protocol['datasets'].items():
        print('Starting',ds,flush=True)
        data=json.loads(z.read(f'registration/{ds}_rows.json'))
        reg=data['register']; where={v['id']:i for i,v in enumerate(reg)}
        panel=read_np(z,f'registration/{ds}_panel.npz')
        pool_x,pool_p=panel['s'],panel['p']
        orig=read_np(z,f'refit_inputs/{ds}.npz')
        anchor=orig['register_ids'].tolist()
        ix=[where[x] for x in anchor]
        np.testing.assert_allclose(pool_x[ix],orig['s'],atol=1e-12)
        np.testing.assert_allclose(pool_p[ix],orig['p'],atol=1e-12)
        pool_x[ix],pool_p[ix]=orig['s'],orig['p']
        queries=data['queries'].copy()
        assert [r['id'] for r in queries]==spec['cohorts'][ds]['ids']
        inputs=[orig['t']]; obs=[orig['pq']]
        if ds=='spider':
            conf=read_np(z,'refit_inputs/confirmation.npz')
            gold=json.loads(z.read('registration/confirmation_gold.json'))
            assert list(conf['ids'])==spec['cohorts']['confirmation']['ids']
            queries += [dict(id=i,group=g,new=gold[i]) for i,g in zip(conf['ids'],conf['query_groups'],strict=True)]
            inputs.append(conf['t']);obs.append(conf['pq'])
        t,pq=np.vstack(inputs),np.vstack(obs)
        all_groups=np.array([r['group'] for r in queries]); ids=np.array([r['id'] for r in queries])
        original_rows=[reg[where[v]] for v in anchor]
        orig_sets={tuple(sorted(r['new'])) for r in original_rows}
        orig_atoms={s for r in original_rows for s in r['new']}
        original_unseen=np.array([tuple(sorted(r['new'])) not in orig_sets and set(r['new'])<=orig_atoms for r in queries])
        for budget in spec['budgets']:
            endpoint=budget in [75,spec['n']]
            for seed in ([1] if endpoint else range(1,6)):
                selected=spec['paths'][str(seed)][:budget]
                # Full pool is the same training set for all orders. Canonicalize row order.
                if budget==spec['n']: selected=anchor+sorted(set(selected)-set(anchor))
                take=[where[k] for k in selected]
                train=[reg[i] for i in take]
                labels=sorted({s for r in train for s in r['new']})
                old=indicators(train,'old',sorted({s for r in train for s in r['old']}))
                y=indicators(train,'new',labels);groups=np.array([r['group'] for r in train])
                if budget==75:
                    assert labels==orig['labels'].tolist()
                    np.testing.assert_array_equal(y,orig['y']);np.testing.assert_array_equal(old,orig['a'])
                predictions,tau=classify(api,pool_x[take],t,pool_p[take],pq,old,y,groups)
                vocab={s:1<<i for i,s in enumerate(labels)}
                truth=np.array([sum(vocab[s] for s in r['new']) if set(r['new'])<=vocab.keys() else -2 for r in queries])
                known=set(y.astype(int) @ (1<<np.arange(len(labels))))
                curr=(truth>=0)&~np.isin(truth,list(known))
                vocabs.append(dict(dataset=ds,budget=budget,path=seed,n_groups=len(set(groups)),n_sets=len(known),labels=labels,threshold=tau))
                df=pd.DataFrame(dict(id=ids,group=all_groups,budget=budget,path=seed,gold=truth,orig_unseen=original_unseen,own_unseen=curr,**predictions))
                for tag,part in spec['cohorts'].items():
                    scope=df[df.id.isin(part['ids'])].copy()
                    assert scope.id.tolist()==part['ids']
                    scope['dataset']=tag
                    snapshots.append(scope)
                    if budget==75:
                        prior=pd.read_csv(OUT.parent/'direct_b_service/requests.csv',dtype={'id':str})
                        prior=prior.query('dataset==@tag').set_index('id').loc[scope.id]
                        for cur,prev in [('direct','b_only'),('direct_cv','b_only_cv'),('forward','forward'),('inverse','inverse'),('gold','gold')]:
                            np.testing.assert_array_equal(scope[cur],prior[prev])
                        np.testing.assert_array_equal(scope.orig_unseen,prior.stratum.eq('unseen_combination'))
                    for stratum,sub in [('all',scope),('fixed_unseen',scope[scope.orig_unseen]),('own_unseen',scope[scope.own_unseen])]:
                        base=dict(dataset=tag,budget=budget,path=seed,stratum=stratum,n=len(sub))
                        ok={m:sub[m].eq(sub.gold) for m in predictions}
                        rows.append(dict(**base,**{m:int(h.sum()) for m,h in ok.items()}))
                        for a,b in [('forward','direct'),('inverse','forward'),('inverse','direct'),('inverse','direct_cv'),('forward','direct_cv')]:
                            diff=ok[a].astype(int)-ok[b].astype(int)
                            pair=dict(**base,method=a,reference=b,fixes=int((diff==1).sum()),harms=int((diff==-1).sum()),delta_pp=100*diff.mean())
                            changes.append(pair)
                            if endpoint and len(sub):
                                codes,levels=pd.factorize(sub.group,sort=True)
                                draws=np.random.default_rng(251121).multinomial(len(levels),np.full(len(levels),1/len(levels)),size=2000)
                                stats=(draws@np.bincount(codes,weights=diff))*100/(draws@np.bincount(codes))
                                lo,hi=np.quantile(stats,[.025,.975])
                                intervals.append(dict(**pair,lower=lo,upper=hi))
            print(ds,budget,'done',flush=True)
    pd.concat(snapshots).to_csv(OUT/'predictions.csv.gz',index=False)
    for name,values in [('hits',rows),('changes',changes),('endpoint_intervals',intervals)]:
        pd.DataFrame(values).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'fits.json').write_text(json.dumps(vocabs,ensure_ascii=False,indent=1),encoding='utf8')
    res=pd.DataFrame(rows)
    for m in ['direct','direct_cv','forward','inverse']: res[m]=100*res[m]/res.n
    res['inverse_minus_direct']=res.inverse-res.direct
    res['forward_minus_direct']=res.forward-res.direct
    res['inverse_minus_forward']=res.inverse-res.forward
    res.groupby(['dataset','stratum','budget']).agg(paths=('path','size'),mean_n=('n','mean'),direct=('direct','mean'),direct_cv=('direct_cv','mean'),forward=('forward','mean'),inverse=('inverse','mean'),forward_minus_direct=('forward_minus_direct','mean'),inverse_minus_forward=('inverse_minus_forward','mean'),inverse_minus_direct=('inverse_minus_direct','mean'),gap_min=('inverse_minus_direct','min'),gap_max=('inverse_minus_direct','max')).reset_index().to_csv(OUT/'curve.csv',index=False)

if __name__=='__main__':
    with ZipFile(BASE) as z, tempfile.TemporaryDirectory() as tmp, threadpool_limits(2):
        path=Path(tmp)/'refit_draws.py'; path.write_bytes(z.read('round2_draws/refit_draws.py'))
        spec=importlib.util.spec_from_file_location('pinned_draws',path)
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        run(z,api)
