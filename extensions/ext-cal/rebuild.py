"""Reconstruct exact-hit registration selectors and frozen test grid decisions."""
from pathlib import Path
from argparse import ArgumentParser
import hashlib,json
import numpy as np
import pandas as pd
GRID=np.array([1.,.5,2.,.25,0.])

def build(ref,jobs,scores,sealed,out,method):
    out.mkdir(parents=True,exist_ok=True)
    saved=[]; led=[]
    raw=pd.read_csv(ref/'decoder/results/loss_inputs.csv.gz',dtype={'id':str})
    for task in ['cn','spider']:
        with np.load(ref/'primary'/f'{task}.npz') as c: original=dict(c)
        labelled=list(original['labels'])
        old=raw[(raw.task==task)&(raw.arm=='full')].groupby('id')
        for f in range(5):
            name=f'{task}-{f}'
            with np.load(scores/f'{method}_{name}.npz') as s,np.load(jobs/f'{name}.npz') as j,np.load(ref/f'cal/folds/{task}_fold{f}.npz') as b:
                np.testing.assert_array_equal(j['ids'],b['held_ids']);np.testing.assert_array_equal(j['reg_ids'],b['train_ids'])
                np.testing.assert_array_equal(j['labels'],b['labels'])
                for k in ('ids','offsets','masks'):np.testing.assert_array_equal(j[k],s[k])
                global_bits=1<<np.array([labelled.index(x) for x in j['labels']])
                local_bits=1<<np.arange(len(global_bits));ids=s['ids']
                ss=s['scores'];ms=s['masks'];off=s['offsets'];bl=b['logits']
                assert np.isfinite(ss).all()
                for i,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
                    v=((ms[lo:hi,None]&local_bits)!=0).astype(float)
                    masks=v.astype(int)@global_bits
                    expected=old.get_group(ids[i])
                    np.testing.assert_array_equal(masks,expected.candidate)
                    e=v@bl[i].astype(float)
                    for mode,pre in [('base',e),('forward',expected.pre.to_numpy())]:
                        preds=masks[(pre[:,None]+ss[lo:hi,None]*GRID).argmax(0)]
                        led.extend((method,task,ids[i],f,mode,w,int(p),int(expected.gold.iloc[0])) for p,w in zip(preds,GRID))
        file=sealed/f'{method}_{task}.npz'
        sha=json.loads((Path(__file__).with_name('sealed.json')).read_text('utf8'))[file.name]
        with file.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sha
        with np.load(file) as s:
            for k in ('ids','offsets','masks'):np.testing.assert_array_equal(original[k],s[k])
            ms=s['masks'];off=s['offsets'];ss=s['scores'];eids=s['ids']
            bits=1<<np.arange(len(labelled))
            for i,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
                opts=ms[lo:hi]; v=((opts[:,None]&bits)!=0).astype(float)
                e=v@original['dvlc'][i].astype(float); bf=e+v@original['forward_logits'][i]
                for mode,pre in [('base',e),('forward',bf)]:
                    preds=opts[(pre[:,None]+ss[lo:hi,None]*GRID).argmax(0)]
                    saved.extend((method,task,eids[i],mode,w,int(p)) for p,w in zip(preds,GRID))
    pd.DataFrame(saved,columns='method task id mode weight pred'.split()).to_csv(out/f'{method}-test.csv.gz',index=False)
    pd.DataFrame(led,columns='method task id fold mode weight pred gold'.split()).to_csv(out/f'{method}-oof.csv.gz',index=False)
if __name__=='__main__':
    p=ArgumentParser()
    for k in ('reference','jobs','oof','sealed','output'):p.add_argument(k,type=Path)
    p.add_argument('method',choices=['chain','scotta'])
    a=p.parse_args();build(a.reference,a.jobs,a.oof,a.sealed,a.output,a.method)
