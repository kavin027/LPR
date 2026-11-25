"""Replay the retrospective 5 by 5 sensitivity and two-prediction overlap."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
GRID=(1.,.5,2.,.25,0.)


def run():
    grid_rows=[]; comparisons=[]; overlaps=[]; selected_rows=[]; choices=[]
    for task in ('cn','spider'):
        with np.load(HERE/(task+'.npz')) as archive:
            z=dict(archive)
        n=len(z['reg'])
        assert n==75 and set(z['fold'])==set(range(5)) and len(z['offs'])==n+len(z['ids'])+1
        def pred(weights,is_registry):
            j,m,i=weights
            start,end=(0,n) if is_registry else (n,len(z['offs'])-1)
            scores=j*z['J']+m*z['M']+i*z['I']
            return np.array([z['masks'][a+np.argmax(scores[a:b])] for a,b in
                zip(z['offs'][start:end],z['offs'][start+1:end+1])])
        cells=[(m,i) for m in GRID for i in GRID]
        hits={pair:int(np.sum(pred((1,*pair),True)==z['rgold'])) for pair in cells}
        winner=max(cells,key=hits.get)
        beta=max(GRID,key=lambda b:hits[b,0])
        seq=(beta,max(GRID,key=lambda a:hits[beta,a]))
        a=max(GRID,key=lambda x:hits[0,x])
        for m,i in cells:
            grid_rows.append((task,m,i,hits[m,i],int((m,i)==winner),int((m,i)==seq)))
        y=z['egold']
        joint=pred((1,*winner),False)
        seq_pred=pred((1,*seq),False);primary=pred((1,0,a),False)
        mapping=pred((0,1,0),False)
        for col,got in [('old_sequential',seq_pred),('old_mapping_only',mapping),('old_primary',primary)]:
            np.testing.assert_array_equal(z[col],got)
        for name,pair in [('joint',winner),('sequential',seq)]:
            selected_rows.append((task,name,*pair,hits[pair],sum(h==max(hits.values()) for h in hits.values())))
        for scope,idx in [('all',np.ones(len(y),dtype=bool)),('unseen',z['unseen'])]:
            jp,sp,truth=joint[idx],seq_pred[idx],y[idx]
            grp=z['groups'][idx]
            names=sorted(set(grp),key=int if task=='cn' else str)
            num=pd.Categorical(grp,categories=names).codes
            ng=len(names); size=np.bincount(num)
            rng=np.random.default_rng(251121)
            w=rng.multinomial(ng,np.full(ng,1/ng),size=2000)
            for ref,ref_pred in [('sequential',sp),('primary',primary[idx]),('mapping_only',mapping[idx])]:
                delta=(jp==truth).astype(int)-(ref_pred==truth).astype(int)
                group_delta=np.bincount(num,weights=delta).astype(int)
                lo,hi=np.percentile(100*(w@group_delta)/(w@size),[2.5,97.5])
                comparisons.append((task,scope,ref,len(truth),int(sum(jp==truth)),int(sum(ref_pred==truth)),
                    int(sum(delta>0)),int(sum(delta<0)),int(sum(jp!=ref_pred)),100*delta.mean(),lo,hi))
            dm,pr=mapping[idx]==truth,primary[idx]==truth
            overlaps.append((task,scope,len(truth),int(sum(dm&pr)),int(sum(dm&~pr)),int(sum(~dm&pr)),
                             int(sum(~dm&~pr)),int(sum(dm|pr)),100*np.mean(dm|pr)))
        for ix,id_ in enumerate(z['ids']):
            choices.append((task,id_,int(y[ix]),int(z['unseen'][ix]),int(joint[ix]),int(seq_pred[ix]),int(primary[ix]),int(mapping[ix])))
    return [('grid',grid_rows,'task beta alpha registry_hits joint_selected sequential_selected'),
        ('weights',selected_rows,'task procedure beta alpha registry_hits tied_joint_maxima'),
        ('paired',comparisons,'task scope reference n joint_hits reference_hits fixes harms changed_sets pg lo hi'),
        ('overlap',overlaps,'task scope n both mapping_only_correct primary_only_correct neither oracle_hits oracle_accuracy'),
        ('decisions',choices,'task id gold unseen joint sequential primary mapping_only')]


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    for name,rows,cols in run():
        df=pd.DataFrame(rows,columns=cols.split())
        path=HERE/(name+('.csv.gz' if name=='decisions' else '.csv'))
        if args.write:df.to_csv(path,index=False)
        else:pd.testing.assert_frame_equal(df,pd.read_csv(path,dtype={'id':str}),rtol=1e-10,atol=1e-10)
    print('PASS: 50 held-out grid cells, 2 registration selections, original comparators and all 4,711 decisions.')
