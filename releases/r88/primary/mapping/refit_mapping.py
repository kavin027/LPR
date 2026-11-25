"""Refit every A-to-B ridge and rerank cached candidates using NumPy/pandas alone."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
GRID=np.array([1.,.5,2.,.25,0.])
def mapper(a,y,soft):
    # sklearn Ridge(alpha=1,fit_intercept=True) for dense inputs.
    ax=a-a.mean(0); cy=y-y.mean(0)
    slopes=np.linalg.solve(ax.T@ax+np.eye(ax.shape[1]),ax.T@cy)
    raw=y.mean(0)+(soft-a.mean(0))@slopes
    low=1/(len(y)+2)
    p=raw.clip(low,1-low)
    return np.log(p/(1-p))

def all_scores(t,reference,directory):
    with np.load(directory/'inputs'/f'{t}.npz') as f:
        a,y,soft,ids,reg=(f[k] for k in ('a','y','soft','ids','reg'))
    y=y.astype(float); a=a.astype(float); bits=1<<np.arange(y.shape[1])
    oof=pd.read_csv(reference/'decoder'/'results'/'loss_inputs.csv.gz',dtype={'id':str})
    oof=oof[(oof.task==t)&(oof.arm=='full')]
    gp={id:d for id,d in oof.groupby('id',sort=False)}
    cv=pd.read_csv(directory/'oof.csv.gz',dtype={'id':str})
    cv=cv[cv.task==t]
    key=cv.drop_duplicates('id').set_index('id').loc[reg]
    np.testing.assert_array_equal(y.astype(int)@bits,key.gold)
    obs={}
    for fold in range(5):
        h=(key.fold==fold).to_numpy()
        old=a[~h].any(0); labels=y[~h].any(0)
        m=mapper(a[~h][:,old],y[~h][:,labels],soft[:len(reg)][h][:,old])
        bs=bits[labels]
        counts=np.log1p(np.bincount(y[~h].sum(1).astype(int),minlength=y.shape[1]+1))
        for i,id in enumerate(reg[h]):
            d=gp[id]
            masks=d.candidate.to_numpy()
            v=((masks[:,None]&bs)!=0).astype(float)
            np.testing.assert_array_equal(v.astype(int)@bs,masks)
            obs[id]=(masks,d.pre.to_numpy(),v@m[i],d.inverse.to_numpy(),counts[v.sum(1).astype(int)])
    w=pd.read_csv(directory/'weights.csv').set_index('task').loc[t]
    for id,(mask,j,m,inv,c) in obs.items():
        scores=dict(map=j[:,None]+m[:,None]*GRID, primary=j[:,None]+inv[:,None]*GRID,
                    map_inverse=(j+w['map']*m)[:,None]+inv[:,None]*GRID,
                    map_count=(j+w['map']*m)[:,None]+c[:,None]*GRID)
        d=cv[cv.id==id]
        for rule,s in scores.items():
            original=d[d.rule==rule].set_index('weight').loc[GRID].pred
            np.testing.assert_array_equal(mask[s.argmax(0)],original)
    m=mapper(a,y,soft[len(reg):])
    counts=np.log1p(np.bincount(y.sum(1).astype(int),minlength=y.shape[1]+1))
    original=pd.read_csv(directory/'predictions.csv.gz',dtype={'id':str})
    original=original[original.task==t].set_index('id').loc[ids]
    methods=['unary','map_only','primary','map','map_inverse','map_count']
    obtained=[]
    with np.load(reference/'primary'/f'{t}.npz') as r:
        np.testing.assert_array_equal(ids,r['ids'])
        masks,offs=r['masks'],r['offsets']; un=r['forward_logits']+r['dvlc'].astype(float)
        g,lin=r['gram'],r['linear']
        for i,(beg,end) in enumerate(zip(offs[:-1],offs[1:])):
            opts=masks[beg:end]
            v=((opts[:,None]&bits)!=0).astype(float); np.testing.assert_array_equal(v.astype(int)@bits,opts)
            j=v@un[i]; ms=v@m[i]; inv=v@lin[i]-.5*((v@g)*v).sum(1)
            msj=j+w['map']*ms
            scores=[j,ms,j+w.primary*inv,msj,msj+w.map_inverse*inv,msj+w.map_count*counts[v.sum(1).astype(int)]]
            obtained.append([opts[s.argmax()] for s in scores])
    np.testing.assert_array_equal(obtained,original[methods].to_numpy())
    archived=pd.read_csv(reference/'interface'/'predictions.csv.gz',dtype={'id':str})
    archived=archived[archived.task==t].set_index('id').loc[ids]
    for new,old in [('map_only','mapping'),('unary','base_forward'),('primary','full_selected')]:
        np.testing.assert_array_equal(original[new],archived[old])

def run(reference,directory):
    for tag in ('cn','spider'):all_scores(tag,reference,directory)
    print('PASS: refitted all 12 registry A-to-B mappings; all fold predictions and 4,711 six-arm query decisions match.')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,default=HERE.parent)
    parser.add_argument('--directory',type=Path,default=HERE)
    args=parser.parse_args();run(args.reference,args.directory)
