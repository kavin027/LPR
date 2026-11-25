"""Fit the text-only target head from inner products and decode every original proposal."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
GRID=(1.,.5,2.,.25,0.)

def logits(kernel,cross,targets):
    # K_c=H K H. The new row is centered against the training rows only.
    rowmean=kernel.mean(0)
    kc=kernel-rowmean[:,None]-rowmean[None,:]+rowmean.mean()
    kq=cross-cross.mean(1,keepdims=True)-rowmean+rowmean.mean()
    pred=targets.mean(0)+kq@np.linalg.solve(kc+np.eye(len(kc)),targets-targets.mean(0))
    low=1/(len(kc)+2);p=pred.clip(low,1-low)
    return np.log(p/(1-p))

def replay(reference,folder):
    import sys
    oof_all=pd.read_csv(reference/'decoder'/'results'/'loss_inputs.csv.gz',dtype={'id':str})
    truth=pd.read_csv(reference/'interface'/'predictions.csv.gz',dtype={'id':str,'group':str})
    saved=[];select=[];foldrows=[]
    for tag in ('cn','spider'):
        with np.load(folder/'inputs'/f'{tag}.npz') as d:
            k,kq,y,reg,ids,labels,fold,forward=(d[s] for s in ('kernel','query_kernel','targets','reg','ids','labels','fold','forward'))
        bits=1<<np.arange(y.shape[1]); reg_gold=y.astype(int)@bits
        src=oof_all[(oof_all.task==tag)&(oof_all.arm=='full')]
        parts={id:sub for id,sub in src.groupby('id',sort=False)}
        totals=np.zeros(len(GRID),int)
        for j in range(5):
            h=fold==j;keep=y[~h].any(0);bs=bits[keep]
            with np.load(reference/'cal'/'folds'/f'{tag}_fold{j}.npz') as b:
                np.testing.assert_array_equal(reg[h],b['held_ids'])
                np.testing.assert_array_equal(reg[~h],b['train_ids'])
                np.testing.assert_array_equal(b['labels'],labels[keep])
            text=logits(k[np.ix_(~h,~h)],k[np.ix_(h,~h)],y[np.ix_(~h,keep)].astype(float))
            for t,i in enumerate(np.flatnonzero(h)):
                row=parts[reg[i]]
                np.testing.assert_array_equal(np.unique(row.gold),[reg_gold[i]])
                opts=row.candidate.to_numpy();v=((opts[:,None]&bs)!=0).astype(float)
                np.testing.assert_array_equal(v.astype(int)@bs,opts)
                pre=v@(text[t]+forward[i,keep]);inc=row.inverse.to_numpy()
                wins=opts[(pre[:,None]+inc[:,None]*GRID).argmax(0)]
                totals+=wins==reg_gold[i]
                foldrows.extend((tag,reg[i],j,w,int(reg_gold[i]),int(pr)) for w,pr in zip(GRID,wins))
        alpha=GRID[totals.argmax()]
        select.extend((tag,w,h,w==alpha) for w,h in zip(GRID,totals))
        text=logits(k,kq,y.astype(float))
        test=truth[truth.task==tag].set_index('id').loc[ids]
        with np.load(reference/'primary'/f'{tag}.npz') as c:
            np.testing.assert_array_equal(ids,c['ids']);np.testing.assert_array_equal(labels,c['labels'])
            masks,offs,unary,g,linear=c['masks'],c['offsets'],c['forward_logits'],c['gram'],c['linear']
            for i,(lo,hi) in enumerate(zip(offs[:-1],offs[1:])):
                opts=masks[lo:hi];v=((opts[:,None]&bits)!=0).astype(float)
                np.testing.assert_array_equal(v.astype(int)@bits,opts)
                e=v@text[i];bf=v@(text[i]+unary[i]);inv=v@linear[i]-.5*((v@g)*v).sum(1)
                winners=[int(opts[s.argmax()]) for s in (e,bf,bf+alpha*inv)]
                r=test.iloc[i]
                saved.append((tag,ids[i],r.group,int(r.gold),r.stratum,*winners))
    forecasts=pd.DataFrame(saved,columns='task id group gold stratum base forward inverse'.split())
    fs=pd.DataFrame(foldrows,columns='task id fold weight gold pred'.split())
    cv=pd.DataFrame(select,columns='task weight hits selected'.split())
    return forecasts,fs,cv

def metrics(pred):
    out=[]
    for tag,b in pred.groupby('task'):
        for cohort,d in [('all',b),('unseen',b[b.stratum=='unseen_combination']),('seen',b[b.stratum=='seen'])]:
            flags={m:d[m].eq(d.gold).to_numpy() for m in ('base','forward','inverse')}
            diff=flags['inverse'].astype(int)-flags['forward'].astype(int)
            gs=sorted(d.group.unique(),key=int if tag=='cn' else str)
            code=pd.Categorical(d.group,categories=gs).codes
            sz=np.bincount(code);n=len(gs)
            boot=np.random.default_rng(251121).multinomial(n,np.full(n,1/n),2000)
            lo,hi=np.percentile(100*(boot@np.bincount(code,weights=diff))/(boot@sz),[2.5,97.5])
            out.append((tag,cohort,len(d),*[int(h.sum()) for h in flags.values()],*[100*h.mean() for h in flags.values()],int((diff>0).sum()),int((diff<0).sum()),100*diff.mean(),lo,hi))
    return pd.DataFrame(out,columns='task cohort n base_hits forward_hits inverse_hits base_acc forward_acc inverse_acc fixes harms gain lo hi'.split())

def run(reference,folder,write):
    pred,folds,cv=replay(reference,folder)
    res=metrics(pred)
    tables={'predictions.csv.gz':pred,'oof.csv.gz':folds,'weights.csv':cv,'summary.csv':res}
    for name,frame in tables.items():
        p=folder/name
        if write:frame.to_csv(p,index=False)
        else:pd.testing.assert_frame_equal(frame,pd.read_csv(p,dtype={'id':str,'group':str}),atol=1e-10,rtol=1e-10)
    print(res.round(3).to_string(index=False))
    print('PASS: both same-encoder heads refit in each original fold and at deployment; pooled weights and paired statistics recomputed.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,default=HERE.parent)
    ap.add_argument('--directory',type=Path,default=HERE);ap.add_argument('--write',action='store_true')
    a=ap.parse_args();run(a.reference,a.directory,a.write)
