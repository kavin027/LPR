"""Batch relabeling workload and frozen score-interface controls.
Run: python workload.py EXTRACTED_ARCHIVE REBUILT_JSONL NEW_OUTPUT
"""
import argparse
import json
import os
import platform
import sqlite3
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from scipy.linalg import solve
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_info, threadpool_limits


def fit_ridge(x, y):
    xm, ym = x.mean(0), y.mean(0)
    xc = x - xm
    dual = solve(xc @ xc.T + np.eye(len(x)), y - ym, assume_a="pos")
    return xc.T @ dual, xm, ym


def predict(model, x):
    w, xm, ym = model
    return ym + (x - xm) @ w


def grouped(x,y,groups):
    result = np.empty((len(x),y.shape[1]))
    for g in np.unique(groups):
        keep = groups != g
        result[~keep] = predict(fit_ridge(x[keep],y[keep]),x[~keep])
    return result


def fit(data,p,arm):
    y, groups = data['y'], data['groups']
    cal = fit_ridge(p,data['a'])
    features = np.column_stack((data['s'],p,grouped(p,data['a'],groups).clip(0,1)))
    forward = fit_ridge(features,y)
    bits = 1 << np.arange(y.shape[1])
    fit = dict(cal=cal,forward=forward,bits=bits,known=np.unique(y.astype(int)@bits),
               log_prior=np.log1p(np.bincount(y.sum(1).astype(int),minlength=len(bits)+1)),
               flips=((np.arange(32)[:,None]&(1<<np.arange(5)))!=0))
    if arm=='inverse':
        f = fit_ridge(y,p)
        residual = p-grouped(y,p,groups)
        eig,vec = np.linalg.eigh(LedoitWolf().fit(residual).covariance_)
        white = vec/np.sqrt(eig.clip(min=max(1e-12,eig.max()*1e-8)))
        fit.update(white=white, offset=(f[2]-f[1]@f[0])@white, slopes=f[0]@white)
    return fit


def classify(fit,t,pq,arm):
    features=np.column_stack((t,pq,predict(fit['cal'],pq).clip(0,1)))
    probs=predict(fit['forward'],features).clip(1/77,76/77)
    logits=np.log(probs/(1-probs))
    bits=fit['bits']
    nearest=np.argsort(abs(probs-.5),kind='stable',axis=1)[:,:5]
    base=(probs>=.5)@bits
    pools=[]
    for initial,uncertain in zip(base,nearest,strict=True):
        union=np.union1d(fit['known'], initial ^ (fit['flips']@bits[uncertain]))
        pools.append(np.r_[initial,union[union!=initial]])
    masks=np.unique(np.concatenate(pools))
    mat=((masks[:,None]&bits)!=0).astype(float)
    ids={mask:i for i,mask in enumerate(masks)}
    if arm=='inverse':
        profiles=fit['offset']+mat@fit['slopes']
        observations=pq@fit['white']
    sizes=mat.sum(1).astype(int)
    preds=[]
    for i,pool in enumerate(pools):
        ix=np.fromiter((ids[m] for m in pool),int,count=len(pool))
        u=mat[ix]@logits[i]
        if arm=='inverse':
            u-=.5*np.square(profiles[ix]-observations[i]).sum(1)
        else:
            u+=fit['log_prior'][sizes[ix]]
        preds.append(pool[np.argmax(u)])
    return np.array(preds),pools


def materialize(path,ids,masks,bits):
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE assignments (request TEXT PRIMARY KEY, label_mask INTEGER NOT NULL)')
        conn.execute('CREATE TABLE postings (request TEXT NOT NULL, atom INTEGER NOT NULL)')
        conn.executemany('INSERT INTO assignments VALUES (?,?)',((str(k),int(m)) for k,m in zip(ids,masks,strict=True)))
        rows=((str(k),i) for k,m in zip(ids,masks,strict=True) for i,b in enumerate(bits) if m&b)
        conn.executemany('INSERT INTO postings VALUES (?,?)',rows)
        conn.execute('CREATE INDEX by_atom ON postings(atom, request)')
        conn.commit()
    conn.close()


def score_interface(x,kind):
    if kind=='top3':
        idx = np.argsort(-x,kind='stable',axis=1)[:,:3]
        out=np.zeros_like(x)
        np.put_along_axis(out,idx,np.take_along_axis(x,idx,axis=1),axis=1)
        return out
    return (x>=.5).astype(float) if kind=='binary' else x


def load_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]


def lexical_history(rebuilt,tag):
    rows=load_jsonl(rebuilt/f'{tag}_history.jsonl')
    vocab=sorted({label for r in rows for label in r['old']})
    target=np.array([[int(label in r['old']) for label in vocab] for r in rows])
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(2,4),min_df=2)
    x=vectorizer.fit_transform(r['text'] for r in rows)
    clf=Ridge(alpha=1,solver='lsqr',tol=1e-5,max_iter=1000).fit(x,target)
    return vectorizer,clf


def lexical_query(model,rebuilt,tag,role,ids):
    rows=load_jsonl(rebuilt/f'{tag}_{role}.jsonl')
    assert list(ids)==[r['id'] for r in rows]
    return model[1].predict(model[0].transform(r['text'] for r in rows)).clip(0,1)


def metrics(frame):
    rows=[]
    for stratum,part in [('all',frame),('unseen',frame[frame.stratum=='unseen_combination'])]:
        a=part['inverse']==part.gold
        b=part.fixed_size==part.gold
        delta=a.astype(int)-b.astype(int)
        groups=part.assign(d=delta).groupby('group').agg(n=('d','size'),d=('d','sum'))
        if all(str(i).isdigit() for i in groups.index): groups=groups.reindex(sorted(groups.index,key=int))
        k=len(groups)
        draws=np.random.default_rng(251121).multinomial(k,np.full(k,1/k),size=2000)
        lo,hi=np.quantile(100*(draws@groups.d)/(draws@groups.n),[.025,.975])
        rows.append(dict(stratum=stratum,n=len(part),full=int(a.sum()),fixed=int(b.sum()),fixes=int((a&~b).sum()),
                         harms=int((~a&b).sum()),gain_pp=100*delta.mean(),lower_pp=lo,upper_pp=hi,
                         mean_candidates=part.candidate_n.mean(),reachable=int(part.reachable.sum())))
    return rows


def run(root,rebuilt,out):
    out.mkdir(exist_ok=True,parents=True)
    models={}
    stats=[]
    timings=[]
    metadata=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,system=platform.system(),
                  processor=platform.processor(),sqlite=sqlite3.sqlite_version,blas=[])
    for lib in threadpool_info():
        metadata['blas'].append({k:lib.get(k) for k in ('internal_api','version','num_threads','architecture')})
    for tag in ('cn','spider','confirmation'):
        data=dict(np.load(root/'refit_inputs'/f'{tag}.npz'))
        corpus='spider' if tag=='confirmation' else tag
        if corpus not in models:
            models[corpus]=lexical_history(rebuilt,corpus)
        lexical_p=lexical_query(models[corpus],rebuilt,corpus,'register',data['register_ids'])
        role='confirmation' if tag=='confirmation' else 'query'
        lexical_q=lexical_query(models[corpus],rebuilt,corpus,role,data['ids'])
        orig=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'group':str,'id':str})
        assert list(data['ids'])==list(orig.id)
        for kind in ('original','tfidf','top3','binary'):
            p,pq=(lexical_p,lexical_q) if kind=='tfidf' else tuple(score_interface(data[k],kind) for k in ('p','pq'))
            assert p.shape==data['p'].shape and pq.shape==data['pq'].shape
            predictions={}
            for trial in range(3 if kind=='original' else 1):
                arms=['fixed_size','inverse']
                if trial%2: arms.reverse()
                for arm in arms:
                    t0=perf_counter()
                    reg=fit(data,p,arm)
                    t1=perf_counter()
                    pred,pools=classify(reg,data['t'],pq,arm)
                    t2=perf_counter()
                    if kind=='original':
                        with tempfile.TemporaryDirectory() as tmp:
                            f=Path(tmp)/'index.sqlite'
                            materialize(f,data['ids'],pred,reg['bits'])
                            t3=perf_counter()
                            timings.append(dict(dataset=tag,trial=trial,rule=arm,records=len(pred),fit_s=t1-t0,
                                                classification_s=t2-t1,sqlite_s=t3-t2,decode_write_s=t3-t1,
                                                seconds_per_10000=10000*(t3-t1)/len(pred),db_bytes=f.stat().st_size))
                        assert np.array_equal(pred,orig[arm]),(tag,arm)
                    if arm in predictions: assert np.array_equal(predictions[arm],pred)
                    predictions[arm]=pred
            frame=orig[['id','group','stratum','gold']].copy()
            frame=frame.assign(**predictions,candidate_n=list(map(len,pools)),
                               reachable=[g in c for g,c in zip(frame.gold,pools,strict=True)])
            frame.to_csv(out/f'{tag}_{kind}.csv.gz',index=False)
            stats.extend(dict(dataset=tag,interface=kind,**item) for item in metrics(frame))
        print(tag,'completed',flush=True)
    pd.DataFrame(stats).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    (out/'machine.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','rebuilt_jsonl','output'): p.add_argument(k,type=Path)
    args=p.parse_args()
    with threadpool_limits(1): run(args.archive,args.rebuilt_jsonl,args.output)
