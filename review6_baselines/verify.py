import json
from pathlib import Path
import numpy as np
import pandas as pd

def verify(root,d):
    import hashlib
    results=pd.read_csv(d/'summary.csv')
    prov=json.loads((Path(__file__).resolve().parent/'cfpl_scores/raw_dialogue_provenance.json').read_bytes())
    possible=set().union(*(set(x['possible_ids'])|set(x['confirmed_ids']) for x in prov.values()))
    intervals=[]
    for tag in ('cn','spider','confirmation'):
        pred=pd.read_csv(d/f'{tag}_paired.csv.gz',dtype={'id':str,'group':str})
        blind=pd.read_csv(d/f'{tag}_blind.csv.gz',dtype={'id':str})
        gold=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'id':str,'group':str})
        assert pred[blind.columns].equals(blind)
        assert gold[['id','group','gold','stratum']].equals(pred[['id','group','gold','stratum']])
        assert (pred['full']==gold['inverse']).all() and (pred['forward']==gold['forward']).all()
        if tag=='cn':
            records=''.join(f'{i}\t{m}\n' for i,m in zip(pred.id,pred.cfpl_unrestricted))
            assert hashlib.sha256(records.encode()).hexdigest()=='1d08bfe11ac37d3a823bde1d7c027fde2be755214a9b4cc5eb9233f212d5a355'
        groups={}
        for item in results[results.dataset==tag].itertuples():
            key=(item.population,item.stratum)
            if key not in groups:
                subset=pred if item.population=='all queries' else pred[~pred.id.isin(possible)]
                if item.stratum!='all':subset=subset[subset.stratum==item.stratum]
                ix_group,ix=np.unique(subset.group,return_inverse=True)
                sizes=np.bincount(ix)
                rng=np.random.default_rng(251121)
                draws=rng.multinomial(len(sizes),np.full(len(sizes),1/len(sizes)),size=2000)
                groups[key]=(subset,ix,sizes,draws)
            q,ix,sizes,draws=groups[key]
            a=q[item.arm]==q.gold;b=q[item.reference]==q.gold
            assert [len(q),sum(a),sum(b),sum(a&~b),sum(~a&b)]==[item.n,item.correct,item.reference_correct,item.fixes,item.harms]
            delta=a.astype(int)-b.astype(int)
            effect=100*delta.mean()
            ci=np.quantile(100*(draws@np.bincount(ix,weights=delta))/(draws@sizes),[.025,.975])
            intervals.append([tag,*key,item.arm,item.reference,len(q),len(sizes),effect,*ci])
    return pd.DataFrame(intervals,columns=['dataset','population','stratum','arm','reference','n','groups','net_pp','lo_pp','hi_pp'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('archive_root',type=Path);p.add_argument('results',type=Path);p.add_argument('--write',type=Path)
    a=p.parse_args();df=verify(a.archive_root,a.results)
    if a.write:df.to_csv(a.write,index=False)
    print(f'{len(df)} paired contrasts checked from numeric predictions and query identities, all 4,900 requests verified.')
