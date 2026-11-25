"""Candidate-budget sensitivity. Default replay requires only NumPy, pandas and the frozen input."""
from pathlib import Path
import numpy as np
import pandas as pd

def evaluate(frozen,out,ref):
    outcomes=[]
    with np.load(frozen) as a:
        p=a['p'];bits=1<<np.arange(p.shape[1])
        first=(p>=.5)@bits
        near=bits[np.argsort(abs(p-.5),axis=1,kind='stable')]
        unary=a['external'].astype(float)+np.log(p/(1-p))
        known,gram,linear=a['known'],a['gram'],a['linear']
        ids,gold,strata=a['ids'],a['gold'],a['stratum']
        assert len(np.unique(ids))==len(ids)
        for k in (1,3,5,7):
            flips=(np.arange(1<<k)[:,None]&(1<<np.arange(k)))!=0
            for i,name in enumerate(ids):
                opts=np.union1d(known,first[i]^(flips@near[i,:k]))
                opts=np.r_[first[i],opts[opts!=first[i]]]
                v=((opts[:,None]&bits)!=0).astype(float)
                inv=v@linear[i]-.5*((v@gram)*v).sum(1)
                pred=opts[(v@unary[i]+.5*inv).argmax()]
                outcomes.append((k,name,strata[i],len(opts),gold[i],bool(gold[i] in opts),int(pred)))
    q=pd.DataFrame(outcomes,columns=['k','id','stratum','candidates','gold','covered','pred'])
    if ref is not None:
        r=pd.read_csv(ref,dtype={'id':str}).query("task=='spider' and rule=='full'").set_index('id')
        original=q[q.k==5].set_index('id')
        np.testing.assert_array_equal(original.pred,r.loc[original.index].pred)
    c=q.pivot(index='id',columns='k',values='covered').to_numpy(dtype=int)
    assert (np.diff(c,axis=1)>=0).all()
    out.mkdir(parents=True,exist_ok=True)
    q.to_csv(out/'requests.csv.gz',index=False)
    results=[]
    for k,b in q.groupby('k'):
        for cohort,d in [('all',b),('unseen',b[b.stratum=='unseen_combination']),('seen',b[b.stratum=='seen']),('unsupported',b[b.stratum=='unseen_atom'])]:
            n=len(d)
            if n==0:continue
            covered=int(d.covered.sum());hits=int((d.pred==d.gold).sum())
            results.append((k,cohort,n,covered,100*covered/n,hits,100*hits/n,d.candidates.mean(),d.candidates.max()))
    stats=pd.DataFrame(results,columns=['k','cohort','n','covered','coverage_percent','hits','accuracy_percent','mean_candidates','max_candidates'])
    stats.to_csv(out/'summary.csv',index=False)
    print(stats.round(2).to_string(index=False))

if __name__=='__main__':
    from tempfile import TemporaryDirectory
    root = Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        evaluate(root/'inputs.npz',out,root.parent/'decoder/results/predictions.csv.gz')
        for name in ['requests.csv.gz','summary.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(out/name),pd.read_csv(root/'results'/name))
    print('PASS: candidate coverage, exact decisions and primary-pool reference for four budgets.')
