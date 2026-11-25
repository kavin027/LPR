"""Rebuild per-request numerical contrasts using only frozen scores and predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GRID = (0., .05, .1, .15, .25, .5, 1., 2.)

def read(path):
    return pd.read_csv(path, dtype={'id':str,'group':str})

BOOTSTRAPS = {}

def paired(d, left, right):
    a=d[left].eq(d.gold); b=d[right].eq(d.gold)
    delta=a.astype(int)-b.astype(int)
    groups=sorted(d.group.unique(),key=int if d.task.iat[0]=='cn' else str)
    indices=pd.Categorical(d.group,categories=groups).codes
    n=len(groups)
    sizes=np.bincount(indices)
    key=(d.task.iat[0],tuple(groups),tuple(sizes))
    if key not in BOOTSTRAPS:
        w=np.random.default_rng(251121).multinomial(n,np.full(n,1/n),2000)
        BOOTSTRAPS[key]=(w,w@sizes)
    w,den=BOOTSTRAPS[key]
    changes=np.bincount(indices,weights=delta).astype(int)
    lo,hi=np.percentile(100*(w@changes)/den,[2.5,97.5])
    return dict(n=len(d),left_hits=a.sum(),right_hits=b.sum(),fixes=(delta>0).sum(),harms=(delta<0).sum(),pg=100*delta.mean(),lo=lo,hi=hi)

def sets(d):
    yield 'all',d
    yield 'unseen',d[d.stratum=='unseen_combination']
    yield 'seen',d[d.stratum=='seen']

def decode(root,t):
    src=np.load(root/'primary'/f'{t}.npz')
    aux=np.load(HERE/'inputs'/f'{t}.npz')
    np.testing.assert_array_equal(src['ids'],aux['ids'])
    np.testing.assert_array_equal(src['labels'],aux['target_labels'])
    logits={name:src[name].astype(float) for name in ('dvlc','gma','adaptive')}
    logits.update({k:aux[str(k)].astype(float) for k in (251122,251123)})
    logits['standalone']=np.zeros_like(logits['dvlc'])
    alpha,ecp_alpha={'cn':(.25,1.),'spider':(.5,.5)}[t]
    G=src['gram']; L=len(G); bits=1<<np.arange(L)
    mean=G[np.triu_indices(L,1)].mean()
    endpoint=(G.sum(1)-G.diagonal()-(L-1)*mean)/(L-2)
    cnt=np.bincount(aux['registration_sizes'],minlength=L+1)
    blocks=[]
    for i,(left,right) in enumerate(zip(src['offsets'][:-1],src['offsets'][1:])):
        pool=src['masks'][left:right]
        v=((pool[:,None]&bits)!=0).astype(float); k=v.sum(1).astype(int)
        fw=v@src['forward_logits'][i]
        inv=v@src['linear'][i]-.5*((v@G)*v).sum(1)
        row=v@(src['linear'][i]-.5*G.diagonal())-mean*k*(k-1)/2-(k-1)*(v@endpoint)
        for name,lgt in logits.items():
            pre=v@lgt[i]+fw
            mask=lambda s:int(pool[np.argmax(s)])
            item=dict(task=t,base=str(name),id=src['ids'][i],forward=mask(pre),row=mask(pre+alpha*row),full=mask(pre+alpha*inv),ecp=mask(pre+ecp_alpha*np.log1p(cnt[k])),unit_full=mask(pre+inv),unit_row=mask(pre+row),unit_ecp=mask(pre+np.log1p(cnt[k])))
            if name=='dvlc':
                item.update({f'alpha_{a}':mask(pre+a*inv) for a in GRID})
            blocks.append(item)
    return pd.DataFrame(blocks)

def label_metrics(d,pred,labels,unsupported):
    extra=sorted(set('|'.join(unsupported.values()).split('|'))-set(labels)-{''})
    names=list(labels)+extra; bits=1<<np.arange(len(names))
    gold=d.gold.to_numpy().copy()
    for i in np.flatnonzero(gold<0):
        gold[i]=sum(1<<names.index(label) for label in unsupported[d.id.iat[i]].split('|'))
    truth=(gold[:,None]&bits)!=0; out=(d[pred].to_numpy()[:,None]&bits)!=0
    tp=(truth&out).sum(0);den=truth.sum(0)+out.sum(0);present=truth.any(0)
    both=(truth&out).sum(1); either=(truth|out).sum(1)
    return dict(micro=100*2*tp.sum()/den.sum(),macro=100*(2*tp[present]/den[present]).mean(),jaccard=100*np.divide(both,either,out=np.ones(len(both)),where=either!=0).mean())

def run(root):
    meta=read(root/'primary/row_predictions.csv.gz').query("base=='dvlc'")[['task','id','group','stratum','gold']]
    cal=read(root/'cal/predictions.csv.gz').merge(meta,on=['task','id'],validate='one_to_one')
    subs=read(root/'cohort/unsupported_targets.csv')
    placebo=read(root/'placebo/predictions.csv.gz')
    seeds=read(root/'seeds/paired.csv.gz')
    contrast=[]; transfer=[]; sens=[]; metric=[]; source=[]; original={}
    for task in ('cn','spider'):
        c=cal[cal.task==task]; p=placebo[placebo.task==task]
        z=decode(root,task).merge(meta[meta.task==task],on=['task','id'],validate='many_to_one')
        selected=z[z.base=='dvlc'].set_index('id')
        for param,ref in [('full_selected','full'),('ecp_selected','ecp'),('bf_fixed','forward'),('full_fixed','unit_full'),('row_selected','row')]:
            np.testing.assert_array_equal(c[param],selected.loc[c.id,ref])
        for seed in (251122,251123):
            old=seeds[(seeds.task==task)&(seeds.seed==seed)]
            match=z[z.base==str(seed)].set_index('id').loc[old.id]
            np.testing.assert_array_equal(match.unit_full,old.lpr)
            np.testing.assert_array_equal(match.unit_ecp,old.ecp)
        for pop,d in sets(c):
            for ref in ('bf_fixed','ecp_selected','row_selected','full_fixed'):
                contrast.append(dict(task=task,cohort=pop,reference=ref,**paired(d,'full_selected',ref)))
            for arm in ('bf_fixed','full_fixed','full_selected'):
                u=subs[subs.task==task].set_index('id').labels.to_dict()
                labels=np.load(root/'primary'/f'{task}.npz')['labels']
                metric.append(dict(task=task,cohort=pop,decoder=arm,**label_metrics(d,arm,labels,u)))
        for pop,d in sets(p):
            for a,b in [('legacy_fixed','history_pca_fixed'),('history_pca_cv','base_forward'),('random_projection_cv','base_forward')]:
                source.append(dict(task=task,cohort=pop,left=a,right=b,**paired(d,a,b)))
        for name,b in z.groupby('base'):
            for pop,d in sets(b):
                for ref in ('forward','row','ecp'):
                    transfer.append(dict(task=task,base=name,cohort=pop,reference=ref,**paired(d,'full',ref)))
                if name=='dvlc':
                    for a in GRID:sens.append(dict(task=task,cohort=pop,alpha=a,accuracy=100*d[f'alpha_{a}'].eq(d.gold).mean()))
                if name=='standalone' and pop=='unseen':
                    original[task]=100*(d.unit_full.eq(d.gold).astype(int)-d.unit_row.eq(d.gold)).mean()
    redraw=read(root/'redraw/draws.csv')
    red=[];rank=[]
    for task,orig in original.items():
        d=redraw[(redraw.task==task)&(redraw.base=='standalone')&(redraw.cohort=='as_registered')]
        rank.append(dict(task=task,original=orig,rank=1+(d.pg<orig).sum(),draws=len(d),min=d.pg.min(),max=d.pg.max()))
        for design,sub in d.groupby('mode'):
            red.append(dict(task=task,mode=design,mean=sub.pg.mean(),min=sub.pg.min(),max=sub.pg.max(),wins=(sub.pg>0).sum(),ties=(sub.pg==0).sum(),losses=(sub.pg<0).sum()))
    # For refitted alternative registries the compact archive starts at scored decisions.
    alt=read(HERE/'inputs/transferred_draws.csv.gz')
    prior=read(root/'redraw/predictions.csv.gz')
    keys=['task','mode','seed','base','id']
    old=prior.set_index(keys).sort_index();new=alt.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(old.sort_index(axis=1),new[old.columns].sort_index(axis=1),check_like=True)
    drawres=[]
    for (task,design,seed,base),d in alt.groupby(['task','mode','seed','base']):
        for pop,u in [('all',d),('as_registered',d[d.draw_unseen])]:
            for ref in ('row','ecp'):
                a=u.transferred_full.eq(u.gold); b=u['transferred_'+ref].eq(u.gold)
                drawres.append(dict(task=task,base=base,mode=design,draw=seed,cohort=pop,reference=ref,n=len(u),full_hits=a.sum(),ref_hits=b.sum(),fixes=(a&~b).sum(),harms=(~a&b).sum(),pg=100*(a.astype(int)-b.astype(int)).mean()))
    return {key:pd.DataFrame(value) for key,value in [('contrasts',contrast),('transfers',transfer),('weight_grid',sens),('partial_credit',metric),('observations',source),('unit_redraw',red),('original_rank',rank),('registry_transfers',drawres)]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=HERE.parent)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    for name,d in run(args.root).items():
        path=HERE/'results'/(name+'.csv')
        if args.write:
            d.to_csv(path,index=False)
        else:
            expected=pd.read_csv(path,dtype={'base':str})
            pd.testing.assert_frame_equal(d,expected,check_dtype=False,check_exact=False,rtol=1e-8,atol=1e-8)
    print('PASS: paired cohorts, complete-label metrics, weight sensitivity, fixed-weight observations, seed/base transfers and forty alternative registries.')
if __name__=='__main__':
    main()

