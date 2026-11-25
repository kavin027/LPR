"""Check the distinct lexical readout, aligned label mapping, and within-registry weight rule."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = ('lexical_bf','lexical_selected','lexical_unit','mapping','mapped_oracle','base_forward','full_selected')
def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    data=pd.read_csv(HERE/'predictions.csv.gz',dtype={'id':str,'group':str})
    assert len(data)==4711 and not data.duplicated(['task','id']).any()
    weights=pd.read_csv(HERE/'weights.csv')
    rows=[]
    for task,d in data.groupby('task'):
        block=weights[weights.task==task].reset_index(drop=True)
        assert block.weight.tolist()==[1,.5,2,.25,0]
        best=int(block.hits.idxmax())
        assert block.selected.sum()==1 and block.loc[best,'selected']
        if block.loc[best,'weight']==1:
            np.testing.assert_array_equal(d.lexical_selected,d.lexical_unit)
        baseline=d.lexical_bf
        for cohort,subset in [('all',d),('seen',d[d.stratum=='seen']),('unseen',d[d.stratum=='unseen_combination'])]:
            for method in SCORES:
                before = subset.lexical_bf if method.startswith('lexical_') else subset.base_forward
                hit=subset[method].eq(subset.gold)
                old=before.eq(subset.gold)
                f,h = int((hit&~old).sum()),int((~hit&old).sum())
                rows.append([task,cohort,method,len(subset),int(hit.sum()),100*hit.mean(),f,h,100*(f-h)/len(subset)])
    metrics=pd.DataFrame(rows,columns=['task','cohort','method','n','hits','accuracy','fixes','harms','gain'])
    if args.write:metrics.to_csv(HERE/'metrics.csv',index=False)
    else:
        pd.testing.assert_frame_equal(metrics,pd.read_csv(HERE/'metrics.csv'),check_exact=False,atol=1e-9,rtol=1e-9)
        print('PASS: heterogeneous classifier and mapping accuracies, paired transitions and weight choices.')
if __name__=='__main__':main()

