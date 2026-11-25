"""Recompute grouped weight selections, cohort accuracies and all matched comparisons."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
GRID=[1.,.5,2.,.25,0.]
ARMS=['unary','map_only','primary','map','map_inverse','map_count']
CONTRASTS=[('map','unary'),('map_inverse','map'),('map_inverse','primary'),('primary','map'),('map_count','map'),('map_inverse','map_count'),('map','map_only')]

def read(folder,name):
    return pd.read_csv(folder/name,dtype={'id':str,'group':str})

def verify(folder,write=False):
    led=read(folder,'predictions.csv.gz')
    assert len(led)==len(led.drop_duplicates(['task','id']))==4711
    cv=read(folder,'oof.csv.gz')
    assert len(cv)==2*75*len(GRID)*4
    assert cv.groupby(['task','id']).gold.nunique().eq(1).all()
    assert cv.groupby(['task','group']).fold.nunique().eq(1).all()
    weights=read(folder,'weights.csv').set_index('task')
    counts=[];metrics=[];pairs=[]
    for (task,rule),d in cv.groupby(['task','rule']):
        counts_by_weight=[]
        for w in GRID:
            rows=d[d.weight==w]
            assert len(rows)==rows.id.nunique()==75 and rows.fold.nunique()==5
            counts_by_weight.append(int((rows.pred==rows.gold).sum()))
        chosen=GRID[int(np.argmax(counts_by_weight))]
        assert weights.loc[task,rule]==chosen
        counts.extend((task,rule,w,n,w==chosen) for w,n in zip(GRID,counts_by_weight))
    for task,sub in led.groupby('task'):
        assert len(sub)=={'cn':2744,'spider':1967}[task]
        for scope,d in [('all',sub),('unseen',sub[sub.stratum=='unseen_combination']),('seen',sub[sub.stratum=='seen'])]:
            groups=sorted(d.group.unique(),key=int if task=='cn' else str)
            gnum=pd.Categorical(d.group,categories=groups).codes
            n=len(groups); sz=np.bincount(gnum)
            rng=np.random.default_rng(251121)
            w=rng.multinomial(n,np.full(n,1/n),size=2000)
            den=w@sz
            for arm in ARMS:
                hits=int(d[arm].eq(d.gold).sum())
                metrics.append((task,scope,arm,len(d),hits,100*hits/len(d)))
            for left,right in CONTRASTS:
                delta=d[left].eq(d.gold).astype(int)-d[right].eq(d.gold).astype(int)
                group_delta=np.bincount(gnum,weights=delta).astype(int)
                lo,hi=np.percentile(100*(w@group_delta)/den,[2.5,97.5])
                pairs.append((task,scope,left,right,len(d),int((delta>0).sum()),int((delta<0).sum()),100*delta.mean(),lo,hi))
    for table,rows,cols in [('cv',counts,'task rule weight hits selected'),('accuracy',metrics,'task scope arm n hits accuracy'),
                            ('paired',pairs,'task scope left right n fixes harms pg lo hi')]:
        out=pd.DataFrame(rows,columns=cols.split())
        if write:out.to_csv(folder/(table+'.csv'),index=False)
        else:pd.testing.assert_frame_equal(out,pd.read_csv(folder/(table+'.csv')),rtol=1e-10,atol=1e-10)
    print('PASS: grouped registration weights, cohort accuracies, paired fixes/harms and grouped intervals.')
if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('directory',nargs='?',default=HERE,type=Path)
    args.add_argument('--write',action='store_true')
    args.add_argument('--reference',type=Path,default=HERE.parent)
    args.add_argument('--summary-only',action='store_true')
    a=args.parse_args()
    verify(a.directory,a.write)
    if not a.write and not a.summary_only:
        from refit_mapping import run
        run(a.reference,a.directory)
