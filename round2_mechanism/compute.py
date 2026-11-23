"""Second review replays. Run python compute.py EXTRACTED_ARTIFACT OUTPUT_DIRECTORY."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits


SEEDS = tuple(range(20261122, 20261127))


def ridge_fit(x, z, weights=None):
    if weights is None:
        weights = np.ones(len(x))
    mx,mz = (np.average(v, weights=weights, axis=0) for v in (x,z))
    xc=x-mx
    wx=xc.T*weights
    if xc.shape[0] < xc.shape[1] and np.all(weights==1):
        b=xc.T@np.linalg.solve(xc@xc.T+np.eye(len(xc)),z-mz)
    else:
        b=np.linalg.solve(wx@xc+np.eye(xc.shape[1]),wx@(z-mz))
    return mz-mx@b,b


def predict(train, target, query, weights=None):
    a,b=ridge_fit(train,target,weights)
    return a+query@b


def crossfit(x,z,g,w=None):
    residual=np.empty_like(z,dtype=float)
    for group in np.unique(g):
        test=g==group
        residual[test]=predict(x[~test],z[~test],x[test],None if w is None else w[~test])
    return residual


def geom(y,reg,obs,groups,kind):
    a,b=ridge_fit(y,reg)
    cov=LedoitWolf().fit(reg-crossfit(y,reg,groups)).covariance_
    if kind=='diagonal':
        cov=np.diag(cov.diagonal())
    if kind=='isotropic':
        cov=np.eye(len(cov))*cov.diagonal().mean()
    w,uu=np.linalg.eigh(cov)
    transform=uu/np.sqrt(np.maximum(w,max(1e-12,w.max()*1e-8)))
    bw=b@transform
    return bw@bw.T, ((obs-a)@transform)@bw.T


def flags(s,r,g):
    return int(s.sum()), int((s & ~r).sum()),int((~s & r).sum())


def evaluate(tag, root, out):
    with np.load(root/'refit_inputs'/f'{tag}.npz') as z:
        s,t,p,pq,a,y,g,ids=[z[k] for k in ('s','t','p','pq','a','y','groups','ids')]
    x=np.column_stack([s,p,crossfit(p,a,g).clip(0,1)])
    xt=np.column_stack([t,pq,predict(p,a,pq).clip(0,1)])
    probs=predict(x,y,xt).clip(1/77,76/77)
    logits=np.log(probs/(1-probs))
    dim=p.shape[1]
    pars={kind:geom(y,p,pq,g,kind) for kind in ('full','diagonal','isotropic')}
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        q,r=np.linalg.qr(rng.normal(size=(s.shape[1],dim)),mode='reduced')
        weights=q*np.sign(np.diag(r))
        reg=(s-s.mean(0))@weights
        scale=np.sqrt(p.var(0).mean()/reg.var(0).mean())
        pars[f'random_{seed}']=geom(y,reg*scale,((t-s.mean(0))@weights)*scale,g,'full')
    weights=1/np.maximum(1,y.sum(1))
    intercept,slopes=ridge_fit(y,p,weights)
    errors=p-crossfit(y,p,g,weights)
    variance=max(1e-12,np.mean(np.square(errors)*weights[:,None]))
    bits=1<<np.arange(y.shape[1])
    known=np.unique(y.astype(int)@bits)
    flip=(np.arange(32)[:,None]&(1<<np.arange(5)))!=0
    near=bits[np.argsort(abs(probs-.5),kind='stable')[:,:5]]
    bases=(probs>=.5)@bits
    previous=root/'mechanism'/('new_database_secondary_predictions.csv' if tag=='confirmation' else f'{tag}_predictions.csv')
    prev=pd.read_csv(previous,dtype={'id':str,'group':str})
    gold=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'id':str,'group':str})
    assert ids.tolist()==gold.id.tolist()==prev.id.tolist()
    # Scan all candidates and save count-specific best before reading outcomes.
    outputs=[]; oracles=[]
    for i,uid in enumerate(ids):
        opts=np.union1d(known,bases[i]^(flip@near[i]))
        opts=np.r_[bases[i],opts[opts!=bases[i]]]
        v=((opts[:,None]&bits)!=0).astype(float)
        sizes=v.sum(1).astype(int)
        base=v@logits[i]
        scores={arm:base+v@linear[i]-.5*np.sum((v@G)*v,axis=1) for arm,(G,linear) in pars.items()}
        gmat=pars['full'][0]
        centered=v-y.mean(0)
        scores['registration_mean']=base-.5*np.sum((centered@gmat)*centered,axis=1)
        err=np.square(intercept+v@slopes-pq[i]).sum(1)
        h=np.maximum(1,sizes)
        scores['source_variable_no_det']=base-err/(2*variance*h)
        scores['source_fixed_noise']=base-err/(2*variance)
        scores['forward_convolution']=scores['source_variable_no_det']-.5*dim*np.log(h*variance)
        outputs.append({key:int(opts[score.argmax()]) for key,score in scores.items()})
        oracle={}
        for key,score in scores.items():
            if key.startswith('random'):continue
            oracle[key]={int(k):int(opts[sizes==k][score[sizes==k].argmax()]) for k in np.unique(sizes)}
        oracles.append(oracle)
    df=pd.DataFrame(outputs)
    for key,prev_key in [('full','old'),('diagonal','diagonal'),('isotropic','isotropic'),('forward_convolution','forward_convolution'),(f'random_{SEEDS[0]}','random')]:
        assert (prev[prev_key]==df[key]).all(),(tag,key,(prev[prev_key]!=df[key]).sum())
    for key in ['fixed_size','global_cv','conditional_cv','epcc','independent_smo','pcc_1']:
        df[key]=gold[key]
    for key in ['inverse_only','fixed','unary','convolution','pca']:
        df[key]=prev[key]
    arms=list(df.columns)
    for key in oracles[0]:
        df['oracle_'+key]=[record[key].get(int(n),-2) for record,n in zip(oracles,gold.gold_size)]
    df.insert(0,'gold',gold.gold)
    df.insert(0,'gold_size',gold.gold_size)
    df.insert(0,'stratum',gold.stratum)
    df.insert(0,'group',gold.group)
    df.insert(0,'id',ids)
    df.to_csv(out/f'{tag}_requests.csv',index=False)
    stats=[]; confusion=[]; oracount=[]
    for part,sub in [('all',df),*df.groupby('stratum')]:
        full=sub.full==sub.gold
        for arm in arms:
            best=sub[arm]
            hit=best==sub.gold
            size=best.map(int.bit_count)
            stats.append(dict(dataset=tag,stratum=part,arm=arm,n=len(sub),hits=int(hit.sum()),full_hits=int(full.sum()),full_fixes=int((full&~hit).sum()),full_harms=int((~full&hit).sum()),mean_pred_size=size.mean(),correct_count=int((size==sub.gold_size).sum())))
            if arm.startswith('random_'):continue
            mat=pd.DataFrame(dict(true_size=sub.gold_size,predicted_size=size,correct=hit)).groupby(['true_size','predicted_size']).correct.agg(['size','sum']).reset_index()
            for k,kp,n,c in mat.itertuples(index=False):
                confusion.append(dict(dataset=tag,stratum=part,arm=arm,true_size=k,predicted_size=kp,n=n,exact=c))
        for arm in oracles[0]:
            truth=sub.gold
            full_or=sub.oracle_full==truth
            arm_or=sub['oracle_'+arm]==truth
            for k,ix in [('all',sub.index),*sub.groupby('gold_size').groups.items()]:
                f,r=full_or.loc[ix],arm_or.loc[ix]
                oracount.append(dict(dataset=tag,stratum=part,k=k,arm=arm,n=len(f),same_size_candidate=int((sub.loc[ix,'oracle_'+arm]!=-2).sum()),hits=int(r.sum()),full_hits=int(f.sum()),full_fixes=int((f&~r).sum()),full_harms=int((~f&r).sum())))
    stats=pd.DataFrame(stats)
    mixture=[]
    for arm in arms:
        by=stats[stats.arm==arm].set_index('stratum')
        ds,du=[100*(r.full_hits-r.hits)/r.n for _,r in by.loc[['seen','unseen_combination']].iterrows()]
        q=-ds/(du-ds) if ds != du else np.nan
        mixture.append(dict(dataset=tag,reference=arm,seen_delta_pp=ds,unseen_delta_pp=du,crossover_q=q if 0<=q<=1 else np.nan,identical=ds==du==0))
    return stats,pd.DataFrame(confusion),pd.DataFrame(oracount),pd.DataFrame(mixture)


def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    results=[evaluate(t,root,out) for t in ['cn','spider','confirmation']]
    for name,blocks in zip(['paired','cardinality_confusion','member_ordering','mixture'],zip(*results)):
        pd.concat(blocks,ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    rows=pd.read_csv(out/'confirmation_requests.csv')
    summary=[]
    for g in sorted(rows.group.unique()):
        for part,sub in [('all',rows),('unseen',rows[rows.stratum=='unseen_combination'])]:
            for label,subset in [('within',sub[sub.group==g]),('delete',sub[sub.group!=g])]:
                own,ref=subset.full==subset.gold,subset.fixed_size==subset.gold
                f,h=(own&~ref).sum(),(~own&ref).sum()
                summary.append(dict(database=g,analysis=label,stratum=part,n=len(subset),full=int(own.sum()),size=int(ref.sum()),fixes=f,harms=h,delta_pp=100*(f-h)/len(subset)))
    pd.DataFrame(summary).to_csv(out/'database_deletion.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(2):
        main(args.artifact,args.output)
