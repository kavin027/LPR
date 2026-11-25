"""Verify external coefficients on out-of-fold registration and one frozen test-score grid."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
GRID=[1.,.5,2.,.25,0.]
def read(p):
    return pd.read_csv(p,dtype={'id':str,'group':str})
def run(ref,folder,methods,write=False):
    source=read(ref/'bases/predictions.csv.gz').query("base=='dvlc'")
    impact=[];counts=[]; predictions=[]
    for method in methods:
        reg=read(folder/f'{method}-oof.csv.gz')
        test=read(folder/f'{method}-test.csv.gz')
        for task in ('cn','spider'):
            old=source[source.task==task].set_index('id')
            d=pd.DataFrame(dict(group=old.group,gold=old.gold,stratum=old.stratum,base=old.adapter,forward=old.forward))
            for mode in ('base','forward'):
                cv=reg[(reg.task==task)&(reg['mode']==mode)]
                assert len(cv)==75*len(GRID) and cv.id.nunique()==75 and cv.groupby('id').fold.nunique().eq(1).all()
                hits=[int((c.pred==c.gold).sum()) for c in (cv[cv.weight==w] for w in GRID)]
                alpha=GRID[np.argmax(hits)];counts.extend((method,task,mode,w,n,w==alpha) for w,n in zip(GRID,hits))
                scores=test[(test.task==task)&(test['mode']==mode)].pivot(index='id',columns='weight',values='pred').loc[d.index]
                np.testing.assert_array_equal(scores[0.],d[mode])
                if mode=='base':np.testing.assert_array_equal(scores[1.],old[method])
                d[mode+'_unit']=scores[1.]
                d[mode+'_selected']=scores[alpha]
            predictions.append(d.assign(method=method,task=task).reset_index())
            for name,q in [('all',d),('unseen',d[d.stratum=='unseen_combination']),('seen',d[d.stratum=='seen'])]:
                groups=sorted(q.group.unique(),key=int if task=='cn' else str); ix=pd.Categorical(q.group,categories=groups).codes
                n=len(groups); counts_per_group=np.bincount(ix)
                boot=np.random.default_rng(251121).multinomial(n,np.full(n,1/n),2000)
                for mode in ('base','forward'):
                    for rule in ('unit','selected'):
                        pred=q[mode+'_'+rule]; delta=pred.eq(q.gold).astype(int)-q[mode].eq(q.gold).astype(int)
                        lo,hi=np.percentile(100*(boot@np.bincount(ix,weights=delta))/(boot@counts_per_group),[2.5,97.5])
                        impact.append((method,task,name,mode,rule,len(q),int(pred.eq(q.gold).sum()),100*pred.eq(q.gold).mean(),int((delta>0).sum()),int((delta<0).sum()),100*delta.mean(),lo,hi))
    stats=pd.DataFrame(impact,columns='method task cohort mode rule n hits accuracy fixes harms gain lo hi'.split())
    choices=pd.DataFrame(counts,columns='method task mode weight hits selected'.split())
    ledger=pd.concat(predictions,ignore_index=True)
    for name,table in [('summary',stats),('choices',choices),('predictions',ledger)]:
        path=folder/(name+('.csv.gz' if name=='predictions' else '.csv'))
        if write:table.to_csv(path,index=False)
        else:pd.testing.assert_frame_equal(table,read(path),atol=1e-10,rtol=1e-10)
    print(choices[choices.selected].to_string(index=False))
    print(stats[stats.cohort.isin(['all','unseen'])].round(2).to_string(index=False))
    print('PASS: registry weights and all blind grid outcomes verified; unit-weight and zero-weight historical masks agree.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',default=['chain','scotta']);p.add_argument('--reference',type=Path,default=HERE.parent)
    p.add_argument('--directory',type=Path,default=HERE);p.add_argument('--write',action='store_true');p.add_argument('--no-score-rebuild',action='store_true')
    a=p.parse_args()
    if not a.write and not a.no_score_rebuild:
        from tempfile import TemporaryDirectory
        from rebuild import build
        with TemporaryDirectory() as scratch:
            scratch=Path(scratch)
            for m in a.methods:
                build(a.reference,a.directory/'roles',a.directory/'scores',a.directory/'sealed',scratch,m)
                for split in ('test','oof'):
                    key=m+'-'+split+'.csv.gz'
                    pd.testing.assert_frame_equal(read(scratch/key),read(a.directory/key),check_exact=True)
        print('PASS: all grids rebuilt from original sealed whole-candidate scores, fold-role masks and neural OOF scores.')
    run(a.reference,a.directory,a.methods,a.write)
