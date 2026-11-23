"""Arbitrary-width candidate masks, sparse member sums, and a decoder-only scale audit."""
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from workload import fit_ridge, grouped, predict


def mask_for(indexes):
    return sum(1 << int(i) for i in indexes)


def members(mask):
    found=[]
    while mask:
        low=mask & -mask
        found.append(low.bit_length()-1)
        mask ^= low
    return found


def register(y,s,groups):
    known=sorted({mask_for(np.flatnonzero(row)) for row in y})
    linear=fit_ridge(y,s)
    covariance=LedoitWolf().fit(s-grouped(y,s,groups)).covariance_
    eig, axes=np.linalg.eigh(covariance)
    w=axes/np.sqrt(eig.clip(min=max(1e-12,1e-8*eig[-1])))
    return dict(known=known, intercept=(linear[2]-linear[1]@linear[0])@w,
                slopes=linear[0]@w, white=w,
                sizes=np.log1p(np.bincount(y.sum(1).astype(int),minlength=y.shape[1]+1)))


def decode(reg,probs,observed):
    logit=np.log(probs/(1-probs))
    uncertain=np.argsort(abs(probs-.5),axis=1,kind='stable')[:,:5]
    pools=[]
    for row,idx in zip(probs,uncertain):
        base=mask_for(np.flatnonzero(row>=.5))
        flips=[0]
        for label in idx: flips += [v ^ (1<<int(label)) for v in flips]
        rest=set(reg['known']).union(base^v for v in flips)
        rest.remove(base)
        pools.append([base,*sorted(rest)])
    masks=sorted(set(itertools.chain.from_iterable(pools)))
    indices={m:i for i,m in enumerate(masks)}
    atoms=[members(m) for m in masks]
    profiles=np.stack([reg['intercept']+reg['slopes'][a].sum(0) for a in atoms])
    priors=reg['sizes'][[len(a) for a in atoms]]
    # Row-wise indexed addition only over active members, with a shared zero-padding index.
    width=max(map(len,atoms))
    active=np.full((len(masks),width),probs.shape[1],dtype=int)
    for i,a in enumerate(atoms): active[i,:len(a)]=a
    logits=np.pad(logit,((0,0),(0,1)))
    observed=observed@reg['white']
    full=[]; fixed=[]
    for i,pool in enumerate(pools):
        ix=[indices[m] for m in pool]
        forward=logits[i,active[ix]].sum(1)
        full.append(pool[np.argmax(forward-.5*np.square(profiles[ix]-observed[i]).sum(1))])
        fixed.append(pool[np.argmax(forward+priors[ix])])
    return full,fixed,pools


def check_archive(root):
    from workload import fit, classify
    data=dict(np.load(root/'refit_inputs'/'spider.npz'))
    fit_dense=fit(data,data['p'],'inverse')
    obs=data['pq'][:100]
    q=data['t'][:100]
    feat=np.column_stack((q,obs,predict(fit_dense['cal'],obs).clip(0,1)))
    prob=predict(fit_dense['forward'],feat).clip(1/77,76/77)
    expected=pd.read_csv(root/'controls'/'spider_requests.csv').iloc[:100]
    full,fixed,pools=decode(register(data['y'],data['p'],data['groups']),prob,obs)
    _,dense_pools=classify(fit_dense,q,obs,'inverse')
    assert all(a==b.tolist() for a,b in zip(pools,dense_pools,strict=True))
    assert full==list(expected.inverse) and fixed==list(expected.fixed_size)
    return data


def sample(L, data):
    rng=np.random.default_rng(20261122)
    ns=data['y'].sum(1).astype(int)
    n=len(ns); rows=2000; m=31
    y=np.zeros((n,L))
    for i,count in enumerate(rng.choice(ns,n)): y[i,rng.choice(L,count,replace=False)]=1
    p=rng.random((n,m))
    pq=rng.random((rows,m))
    scores=rng.uniform(1/77,.42,(rows,L))
    for i,count in enumerate(rng.choice(ns,rows)):
        pos=rng.choice(L,count,replace=False)
        scores[i,pos]=rng.uniform(.58,76/77,count)
        uncertain=rng.choice(L,5,replace=False)
        scores[i,uncertain]=.5+np.sign(scores[i,uncertain]-.5)*rng.uniform(.002,.075,5)
    return y,p,scores,pq


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=check_archive(args.archive)
    results=[]
    for L in (50,100,200):
        y,p,scores,pq=sample(L,data)
        for rep in range(3):
            t=perf_counter(); reg=register(y,p,data['groups']);t1=perf_counter()
            full,fixed,pools=decode(reg,scores,pq);t2=perf_counter()
            upper=max(c.bit_length() for pool in pools for c in pool)
            assert upper==L
            results.append(dict(L=L,m=31,n=75,requests=len(scores),repeat=rep,
                                fit_s=t1-t,decode_both_s=t2-t1,ms_per_request=(t2-t1)*1000/len(scores),
                                registry_unique=len(reg['known']),mean_candidates=np.mean(list(map(len,pools))),
                                max_mask_bits=upper))
    pd.DataFrame(results).to_csv(args.output/'scale_timings.csv',index=False)
    print(pd.DataFrame(results).groupby('L')[['fit_s','decode_both_s','ms_per_request','mean_candidates']].median().round(4))


if __name__=='__main__':
    with threadpool_limits(1): main()
