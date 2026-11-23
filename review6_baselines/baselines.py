"""Matched label-powerset, discriminative Ising and CFPL score controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits


def mod(path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def learn_pair(draws,s,p,a,y,g):
    offsets=np.empty_like(y)
    for group in np.unique(g):
        val=g==group
        cols=a[~val].any(0)
        _,_,pr=draws.forward(s[~val],s[val],p[~val],p[val],a[~val][:,cols],y[~val],g[~val])
        offsets[val]=logit(pr)
    x=np.column_stack((s,p))
    mean=x.mean(0)
    u,sigma,vh=np.linalg.svd(x-mean,full_matrices=False)
    keep=sigma>sigma[0]*max(x.shape)*np.finfo(float).eps
    basis=vh[keep].T
    x=np.column_stack(((x-mean)@basis,np.ones(len(x))))
    dim,labels=x.shape[1],y.shape[1]
    left,right=np.triu_indices(labels,1)
    def unpack(theta):
        w=theta[:dim*labels].reshape(dim,labels)
        J=np.zeros((labels,labels))
        J[left,right]=J[right,left]=theta[dim*labels:]
        return w,J
    def obj(theta):
        w,J=unpack(theta)
        f=offsets+x@w+y@J
        d=expit(f)-y
        nll=np.logaddexp(0,(1-2*y)*f).sum()+.5*(theta@theta)
        pair=y.T@d
        grad=np.r_[(x.T@d).ravel(),pair[left,right]+pair[right,left]]+theta
        return nll,grad
    initial=np.zeros(dim*labels+len(left))
    ans=minimize(obj,initial,jac=True,method='L-BFGS-B',options={'maxiter':500})
    assert ans.success,ans.message
    w,J=unpack(ans.x)
    return (mean,basis,w,J),dict(steps=ans.nit,loss=ans.fun,grad_inf=float(max(abs(ans.jac))))


def load_cfpl(folder,ids,reg_ids,labels):
    sig=json.loads((folder/'student_scores_seal.json').read_bytes())
    scores={};reg={}
    for name in ('attraction','hotel','restaurant'):
        f=folder/f'{name}_scores.npz'
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sig[f.name]
        with np.load(f) as z:
            assert np.array_equal(z['labels'],labels)
            assert not set(z['ids'])&scores.keys() and not set(z['reg_ids'])&reg.keys()
            scores.update(zip(z['ids'],z['query']))
            reg.update(zip(z['reg_ids'],z['register']))
    assert set(scores)==set(ids) and set(reg)==set(reg_ids)
    return np.stack([scores[i] for i in ids]), np.stack([reg[i] for i in reg_ids])


def predict(root,tag,folder):
    draws=mod(root/'round2_draws/refit_draws.py')
    mechanism=mod(root/'round2_mechanism/compute.py')
    with np.load(root/'refit_inputs'/f'{tag}.npz') as z:
        s,t,p,pq,a,y,g,ids,reg_ids,labels=[z[k] for k in ('s','t','p','pq','a','y','groups','ids','register_ids','labels')]
    f,z,pr=draws.forward(s,t,p,pq,a,y,g)
    pool=list(draws.options(pr,y))
    bits=1<<np.arange(y.shape[1])
    if tag!='confirmation':
        with np.load(root/'scores'/f'{tag}_candidates.npz') as c:
            assert np.array_equal(c['ids'],ids)
            for opts,lo,hi in zip(pool,c['offsets'][:-1],c['offsets'][1:]):
                assert np.array_equal(opts,c['mask'][lo:hi])
    gram,linear=mechanism.geom(y,p,pq,g,'full')
    known,which=np.unique(y.astype(int)@bits,return_inverse=True)
    x=f-f.mean(0)
    hot=np.eye(len(known))[which]
    dual=np.linalg.solve(x@x.T+np.eye(len(x)),hot-hot.mean(0))
    lp=(z-f.mean(0))@x.T@dual+hot.mean(0)
    similarities=(z/np.linalg.norm(z,axis=1,keepdims=True))@(f/np.linalg.norm(f,axis=1,keepdims=True)).T
    support=np.stack([similarities[:,which==k].max(1) for k in range(len(known))],axis=1)
    (mean,basis,w,J),info=learn_pair(draws,s,p,a,y,g)
    unary=np.column_stack(((np.column_stack((t,pq))-mean)@basis,np.ones(len(t))))@w
    log_pr=logit(pr)
    if tag=='cn':
        marg,reg_marg=load_cfpl(folder,ids,reg_ids,labels)
        pc=p-p.mean(0);res=y-expit(reg_marg)
        slope=np.linalg.solve(pc.T@pc+np.eye(p.shape[1]),pc.T@(res-res.mean(0)))
        restored=logit(np.clip(expit(marg)+(pq-p.mean(0))@slope,1/77,76/77))
    records=[]
    for i,opts in enumerate(pool):
        v=((opts[:,None]&bits)!=0).astype(float)
        idx=np.flatnonzero(np.isin(opts,known)); ki=np.searchsorted(known,opts[idx])
        ham=np.fromiter(((int(opts[j])^int(opts[0])).bit_count() for j in idx),int)
        tie=np.flatnonzero(ham==ham.min())
        fallback=opts[idx[tie[np.argmax(support[i,ki[tie]])]]]
        u=v@log_pr[i]; inv=v@linear[i]-.5*((v@gram)*v).sum(1)
        row={'id':ids[i],'forward':opts[np.argmax(u)],'full':opts[np.argmax(u+inv)],
             'powerset':opts[idx[np.argmax(lp[i,ki])]],'nearest_registered':fallback,
             'discriminative':opts[np.argmax(u+v@unary[i]+.5*((v@J)*v).sum(1))]}
        if tag=='cn':
            row.update(cfpl=opts[np.argmax(v@marg[i])],cfpl_old=opts[np.argmax(v@restored[i])],
                       cfpl_old_inverse=opts[np.argmax(v@restored[i]+inv)],cfpl_unrestricted=int((marg[i]>=0)@bits))
        records.append(row)
    return pd.DataFrame(records),info


def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    folder=Path(__file__).resolve().parent/'cfpl_scores'
    prov=json.loads((folder/'raw_dialogue_provenance.json').read_bytes())
    overlapping=set().union(*(set(v['confirmed_ids'])|set(v['possible_ids']) for v in prov.values()))
    stats=[];fits={}
    for tag in ('cn','spider','confirmation'):
        pred,fits[tag]=predict(root,tag,folder)
        pred.to_csv(out/f'{tag}_blind.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        base=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'id':str,'group':str})
        assert pred.id.tolist()==base.id.tolist() and np.array_equal(pred.full,base.inverse) and np.array_equal(pred.forward,base.forward)
        arms=list(pred.columns[1:]); pred['gold']=base.gold;pred['group']=base.group;pred['stratum']=base.stratum
        subsets={'all queries':pred}
        if tag=='cn':
            assert len(overlapping & set(pred.id))==325
            subsets['overlap exclusion']=pred[~pred.id.isin(overlapping)]
        for population,cohort in subsets.items():
            for part,chunk in [('all',cohort),*cohort.groupby('stratum')]:
                for arm in arms:
                    for reference in (['full','cfpl_old'] if arm=='cfpl_old_inverse' else ['full']):
                        hit=chunk[arm]==chunk.gold;b=chunk[reference]==chunk.gold
                        stats.append([tag,population,part,arm,reference,len(chunk),int(hit.sum()),int(b.sum()),int((hit&~b).sum()),int((b&~hit).sum())])
        pred.to_csv(out/f'{tag}_paired.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    table=pd.DataFrame(stats,columns='dataset population stratum arm reference n correct reference_correct fixes harms'.split())
    table.to_csv(out/'summary.csv',index=False)
    (out/'fit.json').write_text(json.dumps(fits,indent=2,default=float),encoding='utf8')
    print(table.query("population=='all queries' and stratum in ['all','unseen_combination'] and reference=='full'").to_string(index=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive_root',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    with threadpool_limits(1):
        run(args.archive_root,args.output)
