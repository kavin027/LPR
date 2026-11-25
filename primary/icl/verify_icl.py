from pathlib import Path
import sys
import pandas as pd
import numpy as np

def read(path):
    return pd.read_csv(path,dtype={'id':str,'group':str})

def run(root, predictions):
    gold=read(root/'primary/row_predictions.csv.gz').query("base=='dvlc'")[['task','id','group','gold','stratum']]
    lpr=read(root/'cal/predictions.csv.gz')[['task','id','full_selected']]
    d=gold.merge(lpr,on=['task','id'],validate='one_to_one').merge(read(predictions),on=['task','id'],validate='one_to_one')
    assert len(d)==len(gold)==len(lpr)==4711
    results=[]
    for task,f in d.groupby('task'):
        with np.load(root/'primary'/f'{task}.npz') as z:
            vals=f.set_index('id').loc[z['ids']]
            assert len(f)==len(vals)
            for i,(l,r) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
                for name in ('registry','retrieved','full_selected'):
                    assert vals[name].iat[i] in z['masks'][l:r]
        for cohort,s in [('all',f),('unseen',f[f.stratum=='unseen_combination'])]:
            groups=sorted(s.group.unique(),key=int if task=='cn' else str)
            g=pd.Categorical(s.group,categories=groups).codes
            w=np.random.default_rng(251121).multinomial(len(groups),np.full(len(groups),1/len(groups)),2000)
            for method in ('registry','retrieved'):
                a=s.full_selected.eq(s.gold);b=s[method].eq(s.gold)
                delta=a.astype(int)-b.astype(int)
                low,high=np.percentile(100*(w@np.bincount(g,weights=delta).astype(int))/(w@np.bincount(g)),[2.5,97.5])
                results.append(dict(task=task,cohort=cohort,reference=method,n=len(s),lpr_hits=a.sum(),icl_hits=b.sum(),lpr_accuracy=100*a.mean(),icl_accuracy=100*b.mean(),fixes=(a&~b).sum(),harms=(~a&b).sum(),gain=100*delta.mean(),low=low,high=high))
    return pd.DataFrame(results)

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    d=run(Path(sys.argv[1]) if len(sys.argv)>1 else here.parent,here/'predictions.csv.gz')
    if '--write' in sys.argv:d.to_csv(here/'paired.csv',index=False)
    else:pd.testing.assert_frame_equal(d,read(here/'paired.csv'),check_exact=False,rtol=1e-9,atol=1e-9)
    print('PASS: both Qwen3-4B constrained ICL predictions lie on each original list and reproduce the same-request paired comparisons.')
