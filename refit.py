"""Portable numerical refit, starting with the provided frozen query/registration arrays."""
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def ridge(train, target, query):
    centered = train-train.mean(0)
    dual = solve(centered@centered.T + np.eye(len(train)),target-target.mean(0),assume_a="pos")
    return target.mean(0)+(query-train.mean(0))@centered.T@dual


def group_predictions(x, y, groups):
    out=np.zeros_like(y,dtype=float)
    for group in np.unique(groups):
        held=groups==group
        out[held]=ridge(x[~held],y[~held],x[held])
    return out


def run(name):
    with np.load(ROOT/'refit_inputs'/f'{name}.npz') as m:
        s,t,p,pq,a,y,g=[m[k] for k in ('s','t','p','pq','a','y','groups')]
    f=np.column_stack((s,p,group_predictions(p,a,g).clip(0,1)))
    z=np.column_stack((t,pq,ridge(p,a,pq).clip(0,1)))
    probs=ridge(f,y,z).clip(1/77,76/77)
    logit=np.log(probs/(1-probs))
    error=p-group_predictions(y,p,g)
    spectrum,vectors=np.linalg.eigh(LedoitWolf().fit(error).covariance_)
    white=vectors/np.sqrt(np.maximum(spectrum,max(1e-12,spectrum.max()*1e-8)))
    bits=1<<np.arange(y.shape[1])
    known=np.unique(y.astype(int)@bits)
    n_k=np.bincount(y.sum(1).astype(int),minlength=y.shape[1]+1)
    flips=(np.arange(32)[:,None]&(1<<np.arange(5)))!=0
    full=[]; size=[]
    allbits=np.arange(1<<y.shape[1])
    matrix=((allbits[:,None]&bits)!=0).astype(float)
    profiles=ridge(y,p,matrix)@white
    for i,prob in enumerate(probs):
        base=int((prob>=.5)@bits)
        nearest=np.argsort(abs(prob-.5),kind="stable")[:5]
        pool=np.union1d(known,base^(flips@bits[nearest]))
        pool=np.r_[base,pool[pool!=base]]
        u=matrix[pool]@logit[i]
        full.append(pool[(u-.5*np.square(profiles[pool]-pq[i]@white).sum(1)).argmax()])
        size.append(pool[(u+np.log1p(n_k[matrix[pool].sum(1).astype(int)])).argmax()])
    import csv
    with (ROOT/'controls'/f'{name}_blind.csv').open(encoding='utf8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert list(map(int,full))==[int(r['inverse']) for r in rows]
    assert list(map(int,size))==[int(r['fixed_size']) for r in rows]
    print(name,len(rows),'identical frozen inverse and size-prior decisions')


if __name__=='__main__':
    with threadpool_limits(2):
        for dataset in ('cn','spider','confirmation'):
            run(dataset)
