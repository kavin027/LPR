"""Reconstruct registration draws, strata and paired effects from per-request predictions."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent/'registration'


def run(tag):
    data=json.loads((D/f'{tag}_rows.json').read_bytes())
    cap=set(json.loads((D/f'{tag}_manifest.json').read_bytes())['cap_ids'])
    index={r['id']:r for r in data['register']}
    selections=json.loads((D/f'{tag}_selections.json').read_bytes())
    with (D/f'{tag}_draws.csv').open(encoding='utf-8-sig') as f:
        reported={(int(r['seed']),r['mode'],int(r['budget']),float(r['alpha']),r['stratum']):r for r in csv.DictReader(f)}
    with np.load(D/f'{tag}_predictions.npz') as cache:
        for selection in selections:
            seed,mode,n=(selection[k] for k in ('seed','mode','budget'))
            pool=index.keys() if mode=='uniform' else (k for k in index if k in cap)
            ranked=sorted(pool,key=lambda uid:hashlib.sha256(f'dasfaa-registration-v1|{seed}|{uid}'.encode()).hexdigest())
            chosen=ranked[:n]
            assert chosen==selection['ids']
            bits={lab:1<<j for j,lab in enumerate(selection['labels'])}
            registered={sum(bits[v] for v in index[uid]['new']) for uid in chosen}
            truth=np.array([sum(bits[v] for v in row['new']) if all(v in bits for v in row['new']) else -1
                            for row in data['queries']])
            seen=np.isin(truth,list(registered))
            strata={'all':np.ones(len(truth),bool),'seen':seen,'unseen':~seen&(truth>=0),'unsupported':truth<0}
            prefix=f"{seed}-{mode}-{n}-"
            base=cache[prefix+'size']==truth
            for alpha in ([.5,1,2] if n==75 else [1]):
                ours=cache[prefix+str(alpha)]==truth
                for part,mask in strata.items():
                    r=reported[seed,mode,n,alpha,part]
                    vals=[mask.sum(),(base&mask).sum(),(ours&mask).sum(),(ours&~base&mask).sum(),(base&~ours&mask).sum()]
                    assert vals==[int(r[k]) for k in ['n','size_hits','inverse_hits','fixes','harms']]
    print(tag,'verified',len(selections),'registration selections and',len(reported),'stratified results')

if __name__=='__main__':
    for dataset in ('cn','spider'):run(dataset)
