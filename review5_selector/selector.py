"""Group-held-out decoder selection using the anonymous array release."""
import argparse
from itertools import product
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


METRICS = ('full', 'diagonal', 'spherical', 'factor')
ALPHAS = (1., .5, .25, 2.)
BETAS = (0., .25, .5, 1., 2., 4.)
INVERSE = [(m, a, 0.) for m, a in product(METRICS, ALPHAS)]
COUNT = [('none', 0., b) for b in BETAS]
HYBRID = [(m, a, b) for b, m, a in product(BETAS, METRICS, ALPHAS)]
GRID = HYBRID + COUNT


def read(path):
    return json.loads(path.read_bytes())


def module(path):
    spec = importlib.util.spec_from_file_location('draws', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(draws, support, query, configs):
    s, p, old, y, groups = support
    t, pq = query
    _, _, pr = draws.forward(s, t, p, pq, old, y, groups)
    options = list(draws.options(pr, y))
    bits = 1 << np.arange(y.shape[1])
    masks = np.unique(np.concatenate(options))
    v = ((masks[:, None] & bits) != 0).astype(float)
    hist = np.bincount(y.sum(1).astype(int), minlength=len(bits)+1)
    size = np.log1p(hist[v.sum(1).astype(int)])
    logit = np.log(pr/(1-pr))
    yc = y-y.mean(0)
    b = np.linalg.solve(yc.T@yc+np.eye(len(bits)), yc.T@(p-p.mean(0)))
    pred, _ = draws.calibration(y, p, groups, y[:0])
    cov = LedoitWolf().fit(p-pred).covariance_
    eig, vec = np.linalg.eigh(cov)
    principal = (vec[:, -3:]*eig[-3:])@vec[:, -3:].T
    covariances = {
        'full':cov, 'diagonal':np.diag(cov.diagonal()),
        'spherical':np.eye(len(cov))*cov.diagonal().mean(),
        'factor': principal + np.diag((cov-principal).diagonal()),
    }
    geom = {}
    for metric in dict.fromkeys(m for m,_,_ in configs if m != 'none'):
        e, u = np.linalg.eigh(covariances[metric])
        w = u / np.sqrt(np.maximum(e, max(1e-12, 1e-8*e.max())))
        bw = b@w
        geom[metric] = (((pq-p.mean(0)+y.mean(0)@b)@w)@bw.T,
                        0.5*(((v@bw)**2).sum(1)))
    keys = list(geom)
    metrics = [keys.index(m)+1 if m in geom else 0 for m,_,_ in configs]
    alpha = np.array([a for _,a,_ in configs])
    beta = np.array([b for _,_,b in configs])
    out = np.empty((len(t),len(configs)),dtype=np.int64)
    for i, opts in enumerate(options):
        positions = np.searchsorted(masks, opts)
        mem = v[positions]
        adjusts = np.column_stack([np.zeros(len(opts))] +
            [mem @ geom[m][0][i] - geom[m][1][positions] for m in keys])
        val = ((mem @ logit[i])[:,None] + adjusts[:,metrics]*alpha +
               size[positions,None]*beta)
        out[i] = opts[np.argmax(val,axis=0)]
    return out


def select(draws, support):
    s,p,a,y,g = support
    wins = np.zeros(len(GRID),dtype=int)
    folds = np.full(len(y),-1)
    for fold, (train, held) in enumerate(GroupKFold(n_splits=5).split(s,groups=g)):
        assert set(g[train]).isdisjoint(g[held])
        folds[held] = fold
        atoms = y[train].any(0)
        yr = y[train][:,atoms]
        ar = a[train][:,a[train].any(0)]
        pred = score(draws,(s[train],p[train],ar,yr,g[train]),(s[held],p[held]),GRID)
        truth = y[held][:,atoms].astype(int) @ (1 << np.arange(yr.shape[1]))
        supported = ~y[held][:,~atoms].any(1)
        wins += ((pred==truth[:,None]) & supported[:,None]).sum(0)
    assert min(folds)==0
    count = dict(zip(GRID,wins))
    result = {}
    for name, cfgs in [('selected',INVERSE),('count_cv',COUNT),('hybrid',HYBRID)]:
        chosen = max(cfgs, key=count.get)
        result[name] = (chosen,int(count[chosen]),sum(count[k]==count[chosen] for k in cfgs))
    return result, folds, wins


def iterate(root,corpus):
    reg=root/'registration'
    with np.load(root/'refit_inputs'/f'{corpus}.npz') as z:
        support=tuple(z[k] for k in ('s','p','a','y','groups'))
        yield 'original',0,list(z['register_ids']),support
    rows=read(reg/f'{corpus}_rows.json')['register']
    idx={r['id']:i for i,r in enumerate(rows)}
    with np.load(reg/f'{corpus}_panel.npz') as z:
        s,p=z['s'],z['p']
    for draw in read(reg/f'{corpus}_selections.json'):
        if draw['budget']!=75:
            continue
        ix=[idx[k] for k in draw['ids']]
        r=[rows[i] for i in ix]
        old=sorted({a for row in r for a in row['old']})
        a=np.array([[k in row['old'] for k in old] for row in r],float)
        y=np.array([[k in row['new'] for k in draw['labels']] for row in r],float)
        g=np.array([row['group'] for row in r])
        yield draw['mode'],draw['seed'],draw['ids'],(s[ix],p[ix],a,y,g)


def run(root,out):
    draws=module(root/'round2_draws/refit_draws.py')
    (out/'predictions').mkdir(parents=True,exist_ok=True)
    summaries, selections, splits, grid, timings=[],[],[],[],[]
    for corpus in ('cn','spider'):
        tags=['cn'] if corpus=='cn' else ['spider','confirmation']
        queries={}
        for tag in tags:
            with np.load(root/'refit_inputs'/f'{tag}.npz') as z:
                queries[tag]=(z['t'],z['pq'],z['ids'].astype(str))
        for mode,seed,ids,support in iterate(root,corpus):
            meta=dict(corpus=corpus,mode=mode,seed=seed)
            start=perf_counter()
            decision, folds, cvhits=select(draws,support)
            cv_seconds=perf_counter()-start
            for uid,group,fold in zip(ids,support[-1],folds):
                splits.append(dict(**meta,id=uid,group=group,fold=int(fold)))
            for (metric,alpha,beta),hits in zip(GRID,cvhits):
                grid.append(dict(**meta,metric=metric,alpha=alpha,beta=beta,cv_hits=int(hits)))
            arms={'fixed_size':('none',0.,1.),'inverse':('full',1.,0.)}
            for name,(cfg,hits,ties) in decision.items():
                metric,alpha,beta=cfg
                selections.append(dict(**meta,method=name,metric=metric,alpha=alpha,beta=beta,
                                       cv_hits=hits,ties=ties,n=len(support[0]),m=support[1].shape[1]))
                arms[name]=cfg
            for tag,(t,pq,query_ids) in queries.items():
                name=f'{tag}-{mode}-{seed}.csv.gz'
                start=perf_counter()
                pred=pd.DataFrame(score(draws,support,(t,pq),list(arms.values())),columns=arms)
                seconds=perf_counter()-start
                timings.append(dict(dataset=tag,mode=mode,seed=seed,n=len(t),selector_seconds=cv_seconds,
                                    combined_query_fit_score_seconds=seconds))
                # Load sealed target outcomes after predictions. Full/fixed are independent identity anchors.
                if mode=='original':
                    ref=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'id':str})
                else:
                    ref=pd.read_csv(root/'round2_draws/predictions'/name,dtype={'id':str})
                assert np.array_equal(query_ids,ref.id),(name,'order')
                for key in ('inverse','fixed_size'):
                    assert np.array_equal(pred[key],ref[key]),(name,key,(pred[key]!=ref[key]).sum())
                if mode=='original':
                    assert np.array_equal(pred.inverse,ref.inverse)
                pred['id']=query_ids
                pred['gold']=ref.gold
                pred['stratum']=ref.stratum.replace('unseen_combination','unseen')
                pred.to_csv(out/'predictions'/name,index=False,compression={'method':'gzip','mtime':0})
                for part,frame in [('all',pred),('unseen',pred[pred.stratum=='unseen'])]:
                    for arm in arms:
                        h=frame[arm]==frame.gold
                        for base in arms:
                            r=frame[base]==frame.gold
                            summaries.append(dict(dataset=tag,mode=mode,seed=seed,stratum=part,n=len(frame),
                                arm=arm,base=base,correct=int(h.sum()),base_correct=int(r.sum()),
                                fixes=int((h & ~r).sum()),harms=int((~h & r).sum()),
                                accuracy=100*h.mean(),net_pp=100*(h.mean()-r.mean())))
            print(corpus,mode,seed,decision,flush=True)
    for name,records in [('selections',selections),('folds',splits),('cv_grid',grid),('paired',summaries),('timings',timings)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    frame=pd.DataFrame(summaries)
    r=frame[frame['mode']!='original']
    r.groupby(['dataset','mode','stratum','arm','base']).agg(
        draws=('seed','size'),mean_acc=('accuracy','mean'),mean_pp=('net_pp','mean'),sd_pp=('net_pp','std'),
        wins=('net_pp',lambda s:(s>0).sum()),ties=('net_pp',lambda s:(s==0).sum()),losses=('net_pp',lambda s:(s<0).sum()),
        fixes=('fixes','sum'),harms=('harms','sum')).reset_index().to_csv(out/'summary.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_root',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.archive_root,args.output_dir)

