"""Replay the paired LPR comparisons from the frozen scores: python reproduce.py"""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def verify():
    with zipfile.ZipFile(ROOT/'paired-evidence.zip') as archive:
        paired=list(csv.DictReader(io.StringIO(archive.read('paired_predictions.csv').decode('utf-8-sig'))))
    for tag in ('cn','spider'):
        rows=[r for r in paired if r['dataset']==tag]
        with np.load(ROOT/'scores'/f'{tag}_candidates.npz') as c, np.load(ROOT/'scores'/f'{tag}_scoring.npz') as par:
            assert [r['request'] for r in rows]==list(c['ids'])==list(par['ids'])
            assert len(set(c['ids']))==len(rows)
            labels=c['labels'].tolist()
            offsets, all_masks, forward, unary, interaction = [c[key] for key in ('offsets','mask','forward','unary','interaction')]
            pcc = par['pcc']
            n=np.bincount(par['register_target'].sum(1),minlength=len(labels)+1)
            hits={k:[] for k in ['forward','inverse','cardinality','pcc']}
            for i,r in enumerate(rows):
                a,b=offsets[i:i+2]
                masks=all_masks[a:b]
                f=forward[a:b]
                size=f+np.log1p(n)[[int(v).bit_count() for v in masks]]
                full=f+unary[a:b]+interaction[a:b]
                pred={'forward':masks[np.argmax(f)],'cardinality':masks[np.argmax(size)],
                      'inverse':masks[np.argmax(full)],'pcc':pcc[i]}
                for arm,mask in pred.items():
                    members={v for j,v in enumerate(labels) if int(mask)&(1<<j)}
                    assert members==set(json.loads(r[arm])),(tag,r['request'],arm)
                    hits[arm].append(members==set(json.loads(r['truth'])))
            unseen=np.array([r['stratum']=='unseen_combination' for r in rows])
            groups=[r['group'] for r in rows if r['stratum']=='unseen_combination']
            keys=sorted(set(groups),key=int if tag=='cn' else str)
            index=[keys.index(g) for g in groups]
            draws=np.random.default_rng(251121).multinomial(len(keys),np.full(len(keys),1/len(keys)),2000)
            for ref in ['forward','cardinality','pcc']:
                delta=(np.asarray(hits['inverse'],int)-np.asarray(hits[ref],int))[unseen]
                intervals=np.quantile(100*(draws@np.bincount(index,weights=delta))/(draws@np.bincount(index)),[.025,.975])
                print(tag,ref,len(delta),sum(delta==1),sum(delta==-1),round(100*delta.mean(),2),intervals.round(2))


if __name__=='__main__':
    verify()
