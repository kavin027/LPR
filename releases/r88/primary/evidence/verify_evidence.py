from pathlib import Path
import sys
import numpy as np
import pandas as pd

def read(f):
    return pd.read_csv(f,dtype={'id':str,'group':str})

def compute(root,out):
    d=read(root/'interface/predictions.csv.gz')
    dec=read(root/'decoder/results/predictions.csv.gz').pivot(index=['task','id'],columns='rule',values='pred').reset_index()
    d=d.merge(dec.drop(columns='base_forward'),on=['task','id'],validate='one_to_one')
    lex=read(out/'lexical-predictions.csv.gz')
    for arm in ('supervised','permuted'):
        x=lex[lex.arm==arm].drop(columns='arm').rename(columns={k:arm+'_'+k for k in ['selected','unit','zero']})
        d=d.merge(x,on=['task','id'],validate='one_to_one')
    assert len(d)==4711
    for a,b in [('supervised_unit','lexical_unit'),('supervised_selected','lexical_selected'),('supervised_zero','lexical_bf'),('permuted_zero','lexical_bf'),('full','full_selected')]:
        np.testing.assert_array_equal(d[a],d[b])
    contrasts=[('full_selected','abc_matched'),('lexical_selected','lexical_bf'),('lexical_unit','lexical_bf'),('lexical_selected','full_selected'),('text_own_inverse','text_own'),('full_selected','base_inverse'),('mapping','base_forward'),('full_selected','mapped_oracle'),('lexical_selected','permuted_selected'),('lexical_unit','permuted_unit')]
    rows=[]
    for task,f in d.groupby('task'):
        for cohort,s in [('all',f),('unseen',f[f.stratum=='unseen_combination'])]:
            groups=sorted(s.group.unique(),key=int if task=='cn' else str)
            g=pd.Categorical(s.group,categories=groups).codes
            w=np.random.default_rng(251121).multinomial(len(groups),np.full(len(groups),1/len(groups)),2000)
            denom=w@np.bincount(g)
            for left,right in contrasts:
                a=s[left].eq(s.gold); b=s[right].eq(s.gold); diff=a.astype(int)-b.astype(int)
                low,high=np.percentile(100*(w@np.bincount(g,weights=diff).astype(int))/denom,[2.5,97.5])
                rows.append((task,cohort,left,right,len(s),100*a.mean(),100*b.mean(),(a&~b).sum(),(~a&b).sum(),100*diff.mean(),low,high))
    return pd.DataFrame(rows,columns='task cohort method reference n accuracy reference_accuracy fixes harms gain low high'.split())
if __name__=='__main__':
    here=Path(__file__).resolve().parent
    result=compute(Path(sys.argv[1]) if len(sys.argv)>1 else here.parent,here)
    if '--write' in sys.argv: result.to_csv(here/'paired.csv',index=False)
    else: pd.testing.assert_frame_equal(result,read(here/'paired.csv'),check_exact=False,rtol=1e-9,atol=1e-9)
    print('PASS: lexical supervision and matched paired controls on 4,711 requests.')
