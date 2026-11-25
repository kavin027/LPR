"""Anonymous numeric replay. Run python verify.py in the extracted directory."""
from pathlib import Path
import hashlib
import json
import os
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    # NumPy reads these at import time. This also supplies a stable default to child scripts.
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(variable, '1')
    def threadpool_limits(limits):
        return nullcontext()

import numpy as np
import pandas as pd

ROOT=Path(__file__).parent
def read(p):return pd.read_csv(ROOT/p,dtype={'id':str,'group':str})
def check(x,y):np.testing.assert_allclose(x,y,atol=2e-7,rtol=1e-9)
def chosen(data,task,cohort,**fields):
    sub=data[data.task.eq(task)]
    for name,value in fields.items():sub=sub[sub[name].eq(value)]
    if cohort!='all':
        tag='unseen_combination' if cohort=='unseen' else cohort
        sub=sub[sub.stratum.eq(tag)]
    return sub
BOOT={}
def paired(d,a,b):
    old=d[b].eq(d.gold);new=d[a].eq(d.gold);delta=new.astype(int)-old.astype(int)
    groups=sorted(d.group.unique(),key=int if d.task.iat[0]=='cn' else str)
    ix=pd.Categorical(d.group,categories=groups).codes;n=len(groups)
    key=(d.task.iat[0],tuple(groups),tuple(np.bincount(ix)))
    if key not in BOOT:
        w=np.random.default_rng(251121).multinomial(n,np.full(n,1/n),2000);BOOT[key]=(w,w@np.bincount(ix))
    w,den=BOOT[key];lo,hi=np.percentile(100*(w@np.bincount(ix,weights=delta))/den,[2.5,97.5])
    return [len(d),old.sum(),new.sum(),(delta>0).sum(),(delta<0).sum(),100*delta.mean(),lo,hi]
def primary():
    ledger=read('primary/row_predictions.csv.gz')
    for t in ('cn','spider'):
        with np.load(ROOT/'primary'/f'{t}.npz') as z:
            G=z['gram']; L=len(G);mu=G[np.triu_indices(L,1)].mean()
            e=(G.sum(1)-G.diagonal()-(L-1)*mu)/(L-2)
            q=z['linear'];u=z['forward_logits'];off=z['offsets'];masks=z['masks'];ids=z['ids'];bit=1<<np.arange(L)
            assert len(ids)==len(set(ids)) and np.all(np.diff(off)>0) and off[-1]==len(masks)
            for base in ('dvlc','gma','adaptive'):
                actual=[];base_score=z[base].astype(float)
                for i,(left,right) in enumerate(zip(off[:-1],off[1:])):
                    pool=masks[left:right];v=((pool[:,None]&bit)!=0).astype(float);k=v.sum(1)
                    pre=v@base_score[i]+v@u[i]
                    fu=pre+v@q[i]-.5*((v@G)*v).sum(1)
                    row=pre+v@(q[i]-.5*G.diagonal())-mu*k*(k-1)/2-(k-1)*(v@e)
                    actual.append([pool[row.argmax()],pool[fu.argmax()]])
                anchor=ledger[(ledger.task==t)&(ledger.base==base)].set_index('id').loc[ids]
                np.testing.assert_array_equal(actual,anchor[['row','full']])
    for r in read('primary/row_metrics.csv').itertuples():
        d=chosen(ledger,r.task,r.subset,base=r.base);n,b,a,f,h,pg,lo,hi=paired(d,'full','row')
        check([n,100*b/n,100*a/n,f,h,pg,lo,hi],[r.n,r.row_acc,r.full_acc,r.fix,r.harm,r.pg,r.lo,r.hi])
    return ledger[ledger.base=='dvlc'][['task','id','group','gold','stratum']]
def optional(key,frame,a,b):
    if not frame.empty:
        n,old,new,f,h,pg,lo,hi=paired(frame,a,b)
        EXTRA.append([*key,n,old,new,f,h,pg,lo,hi])
def all_results(gold):
    key=['task','id'];p=read('placebo/predictions.csv.gz')
    for r in read('placebo/paired.csv').itertuples():
        d=chosen(p,r.task,r.cohort)
        n,b,a,f,h,pg,lo,hi=paired(d,'legacy_'+r.setting,r.comparator)
        check([n,a,b,100*a/n,100*b/n,f,h,pg,lo,hi],[r.n,r.legacy_hits,r.comparator_hits,r.legacy_acc,r.comparator_acc,r.fixes,r.harms,r.pg,r.lo,r.hi])
    for folder,fn in [('additive','blind'),('cal','predictions')]:
        d=read(f'{folder}/{fn}.csv.gz').merge(gold,on=key,validate='one_to_one');assert len(d)==len(gold)
        for r in read(f'{folder}/metrics.csv').itertuples():
            if folder=='cal':
                subset=chosen(d,r.task,r.subset);a='full_'+r.full_setting;b=r.control+'_'+r.control_setting
                n,old,new,f,h,pg,lo,hi=paired(subset,a,b)
                check([n,new,old,100*new/n,100*old/n,f,h,pg,lo,hi],[r.n,r.full_hits,r.ref_hits,r.full_acc,r.ref_acc,r.f,r.h,r.pg,r.lo,r.hi])
            else:
                subset=chosen(d,r.task,r.cohort);a=r.method;b=r.reference
                check(paired(subset,a,b),[r.n,r.before,r.after,r.fix,r.harm,r.pg,r.lo,r.hi])
            if (r.subset if folder=='cal' else r.cohort)=='all':
                optional((folder,r.task,'seen',a,b),chosen(d,r.task,'seen'),a,b)
    for folder in ['seeds','oracle']:
        d=read(f'{folder}/'+('paired.csv.gz' if folder=='seeds' else 'requests.csv.gz'))
        for r in read(f'{folder}/metrics.csv').itertuples():
            if folder=='seeds':
                check(paired(chosen(d,r.task,r.cohort,seed=r.seed),'lpr','ecp'),[r.n,r.ecp_hits,r.lpr_hits,r.fix,r.harm,r.pp,r.lo,r.hi])
            else:
                arms={'full_row':('M5','row'),'oracle_full_row':('oracle_M5','oracle_row'),'specific_full':('M4','M5'),'list_threshold':('adapter','threshold')}
                check(paired(chosen(d,r.task,r.cohort,base=r.base),*arms[r.contrast]),[r.n,r.before,r.after,r.fix,r.harm,r.pg,r.lo,r.hi])
    d=read('bases/predictions.csv.gz')
    for r in read('bases/metrics.csv').itertuples():
        sub=chosen(d,r.task,r.subset,base=r.base);rhit=sub[r.method].eq(sub.gold);lhit=sub.lpr.eq(sub.gold)
        check([len(sub),rhit.sum(),100*rhit.mean(),(lhit&~rhit).sum(),(rhit&~lhit).sum()],[r.n,r.hits,r.accuracy,r.lpr_fixes,r.lpr_harms])
    d=read('rank5/predictions.csv.gz').merge(d[['task','base','id','gold','stratum','group']],on=['task','base','id'],validate='one_to_one')
    for r in read('rank5/metrics.csv').itertuples():
        n,old,new,f,h,pg,lo,hi=paired(chosen(d,r.task,r.subset,base=r.base),'rank5','full')
        check([n,old,new,100*old/n,100*new/n,f,h],[r.n,r.full_hits,r.rank5_hits,r.full_acc,r.rank5_acc,r.fix,r.harm])
    d=read('confirmation/confirmation_requests.csv').rename(columns={'truth':'gold'});d['task']='spider'
    for r in read('confirmation/confirmation_metrics.csv').itertuples():
        if r.n:
            sub=d if r.stratum=='all' else d[d.stratum==r.stratum]
            check(paired(sub,'inverse','size'),[r.n,r.size_hits,r.inverse_hits,r.fixes,r.harms,r.net_pp,r.lo,r.hi])
def grid_checks():
    priority=(1.,.5,2.,.25,0.)
    split=read('cal/folds.csv')
    for task,s in split.groupby('corpus'):
        assert len(s)==len(set(s.id))==75 and s.groupby('group').fold.nunique().max()==1
        for k in range(5):
            h=s.fold.eq(k)
            with np.load(ROOT/'cal/folds'/f'{task}_fold{k}.npz') as z:
                np.testing.assert_array_equal(z['train_ids'],s.id[~h]);np.testing.assert_array_equal(z['held_ids'],s.id[h])
                assert z['logits'].shape==(sum(h),len(z['labels']))
    for folder in ('cal','placebo'):
        f=read(folder+'/oof.csv.gz');name='method' if folder=='cal' else 'arm';pred='mask' if folder=='cal' else 'prediction'
        split.rename(columns={'corpus':'task'},inplace=False)[['task','id','fold']].merge(f[['task','id','fold']].drop_duplicates(),on=['task','id','fold'],validate='one_to_one',how='inner').pipe(lambda z:check(len(z),150))
        f['hit']=f[pred]==f.gold;v=f.groupby(['task',name,'alpha']).hit.agg(['sum','size']).reset_index()
        assert v['size'].eq(75).all()
        v['priority']=v.alpha.map(dict(zip(priority,range(5))))
        v=v.sort_values(['task',name,'sum','priority'],ascending=[True,True,False,True]).drop_duplicates(['task',name])
        b=read(folder+'/selection.csv'); b=b if folder=='cal' else b[b.selected]
        for r in b.to_dict('records'):
            check(v[(v.task==r['task'])&(v[name]==r[name])][['alpha','sum']],[[r['alpha'],r['hits']]])
def draws():
    d=read('redraw/predictions.csv.gz');ex=read('redraw/draws.csv')
    for task in ('cn','spider'):
        meta=json.loads((ROOT/'redraw'/f'{task}_selections.json').read_bytes())
        assert len(meta)==20 and len({(s['seed'],s['mode']) for s in meta})==20
        assert all(len(set(s['ids']))==len(s['ids'])==s['budget']==75 for s in meta)
    for r in ex.itertuples():
        w=d[(d.task==r.task)&(d.base==r.base)&(d.seed==r.seed)&(d['mode']==r.mode)]
        if r.cohort=='as_registered':w=w[w.draw_unseen]
        if r.cohort=='original_unseen':w=w[w.original_unseen]
        a=w.full.eq(w.gold);b=w.row.eq(w.gold);f=(a&~b).sum();h=(b&~a).sum()
        check([len(w),a.sum(),b.sum(),f,h,100*(f-h)/len(w)],[r.n,r.full_hits,r.row_hits,r.fixes,r.harms,r.pg])
    for r in read('redraw/summary.csv').itertuples():
        s=ex[(ex.task==r.task)&(ex.base==r.base)&(ex.cohort==r.cohort)]
        if r.mode!='pooled':s=s[s['mode']==r.mode]
        z=s.pg;check([len(s),s.n.min(),s.n.max(),z.mean(),z.median(),z.std(),z.min(),z.max(),(z>0).sum(),(z==0).sum(),(z<0).sum()],[r.draws,r.n_min,r.n_max,r.mean,r.median,r.sd,r.low,r.high,r.wins,r.ties,r.losses])
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seen-output',type=Path);args=p.parse_args()
    for name,expect in json.loads((ROOT/'manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expect,name
    EXTRA=[]
    with threadpool_limits(limits=1):gold=primary();grid_checks();all_results(gold);draws()
    if args.seen_output:pd.DataFrame(EXTRA,columns=['experiment','task','cohort','method','reference','n','before','after','fix','harm','pg','lo','hi']).to_csv(args.seen_output,index=False)
    print('PASS: 14,133 Row/Full score decisions; 18 cohort-specific primary paired intervals; ten folds, both selection grids; base, calibration, placebo, fitted additive, rank-five, oracle, confirmation, seed and redraw results. 40 standalone and four DVLC registrations.')

    import subprocess, sys
    for extra in ['metric/verify_metrics.py','cohort/verify_cohort.py','common_proposals/verify_common.py','decoder/verify_refits.py','interface/verify_interface.py','interface/verify_mapping.py','stats/verify_statistics.py','candidate_budget/candidate_budget.py','icl/verify_icl.py','registry_votes/verify.py','mapping/verify.py','reselection/verify.py','evidence/verify_evidence.py']:
        subprocess.run([sys.executable,'-B',str(ROOT/extra)],cwd=ROOT,check=True)
