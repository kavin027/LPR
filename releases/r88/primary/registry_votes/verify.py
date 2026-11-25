"""Measure prediction reuse of registered sets, via exact set identity, and cross-method overlap."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def tab(path):
    return pd.read_csv(path,dtype={'id':str,'group':str})

def evaluate(root):
    here=root/'registry_votes'
    icl=tab(root/'icl/predictions.csv.gz')
    mapping=tab(root/'interface/predictions.csv.gz')
    unit=tab(root/'cal/predictions.csv.gz')[['task','id','full_fixed']]
    data=mapping.merge(icl, on=['task','id'],validate='one_to_one').merge(unit,on=['task','id'],validate='one_to_one')
    assert len(data)==len(mapping)==len(unit)==len(icl)==4711
    rows=[];overlap=[];local=[]
    for task,raw in data.groupby('task'):
        reg=tab(here/f'{task}_registry.csv')
        assert len(reg)==reg.id.nunique()==75
        assert set(reg.id).isdisjoint(set(raw.id))
        known=set(reg.set_mask)
        if task=='spider':
            with np.load(root/'candidate_budget/inputs.npz') as z:
                assert known==set(z['known'])
        with np.load(here/f'{task}_retrieved.npz') as z:
            reg_order=pd.DataFrame(z['indices'],index=z['ids']).loc[raw.id].to_numpy()
        assert reg_order.shape==(len(raw),5)
        shown=reg.set_mask.to_numpy()[reg_order]
        assert len(shown)==len(raw)
        raw=raw.copy()
        raw['among_five']=(shown==raw.retrieved.to_numpy()[:,None]).any(axis=1)
        raw['first_demo']=shown[:,0]
        # Resolve ties by the earlier position in the stored retrieval order.
        raw['plurality_demo']=[max(masks,key=lambda x:np.count_nonzero(masks==x)) for masks in shown]
        assert (raw.gold.isin(known)==raw.stratum.eq('seen')).all()
        for cohort,part in [('all',raw),('seen',raw[raw.stratum.eq('seen')]),('unseen',raw[raw.stratum.eq('unseen_combination')])]:
            for arm in ['registry','retrieved','base_forward','mapping','full_selected','full_fixed']:
                given=part[arm].isin(known)
                hit=part[arm].eq(part.gold)
                rows.append([task,cohort,arm,len(part),len(known),int(given.sum()),100*given.mean(),int(hit.sum()),int((hit&~given).sum()),int((~hit&~given).sum())])
            in_all=part.retrieved.isin(known)
            given=part.among_five
            assert not (given&~in_all).any()
            local.append([task,cohort,len(part),int(given.sum()),int((in_all&~given).sum()),int((~in_all).sum()),int(part.retrieved.eq(part.first_demo).sum()),int(part.retrieved.eq(part.plurality_demo).sum())])
            m=part.mapping.eq(part.gold)
            p=part.full_selected.eq(part.gold)
            overlap.append([task,cohort,len(part),int(p.sum()),int(m.sum()),int((p&m).sum()),int((p&~m).sum()),int((~p&m).sum()),int((~p&~m).sum()),int((p|m).sum()),100*(p|m).mean()])
    copies=pd.DataFrame(rows,columns='task cohort arm n registered_sets registered_predictions registered_percent exact_correct correct_unregistered incorrect_unregistered'.split())
    pairs=pd.DataFrame(overlap,columns='task cohort n lpr mapping both lpr_only mapping_only neither either oracle_accuracy'.split())
    retrieved=pd.DataFrame(local,columns="task cohort n shown_among_five other_registered unregistered agrees_first agrees_plurality".split())
    return copies,pairs,retrieved

def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    tables=evaluate(here.parent)
    for name,table in zip(['set_reuse','overlap','retrieved_votes'],tables):
        path=here/f'{name}.csv'
        if args.export:table.to_csv(path,index=False)
        else:pd.testing.assert_frame_equal(tab(path),table)
    print('PASS: registry membership and mapper overlap on both tasks from the 75 dual-annotated requests.')
if __name__=='__main__':main()
