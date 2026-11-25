"""Paired unit-weight DVLC comparison with standalone mapping on unseen SpiderOps requests."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def compare(directory):
    base=pd.read_csv(directory/'primary/bases/predictions.csv.gz',dtype={'id':str,'group':str})
    maps=pd.read_csv(directory/'joint_mapping/decisions.csv.gz',dtype={'id':str})
    left=base[(base.task=='spider') & (base.base=='dvlc') & (base.stratum=='unseen_combination')]
    right=maps[(maps.task=='spider') & maps.unseen.astype(bool)]
    d=left.merge(right,on=['task','id','gold'],validate='one_to_one')
    assert len(left)==len(right)==len(d)==548
    full_hit=d.lpr.eq(d.gold)
    map_hit=d.mapping_only.eq(d.gold)
    diff=full_hit.astype(int)-map_hit.astype(int)
    groups=np.sort(d.group.unique())
    indexes=pd.Categorical(d.group,categories=groups).codes
    bootstrap=np.random.default_rng(251121).multinomial(len(groups),np.repeat(1/len(groups),len(groups)),2000)
    low,high=np.percentile(100*(bootstrap@np.bincount(indexes,weights=diff))/(bootstrap@np.bincount(indexes)),[2.5,97.5])
    return dict(n=len(d),full_hits=int(full_hit.sum()),mapping_hits=int(map_hit.sum()),fixes=int((diff>0).sum()),harms=int((diff<0).sum()),gain=100*diff.mean(),lower=low,upper=high)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('release',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    result=pd.DataFrame([compare(args.release)])
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))
