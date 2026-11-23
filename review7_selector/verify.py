"""Check Round 7 joint selection directly against the frozen grid and prediction rows."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(root, folder):
    old=root/'review5_selector'
    grid=pd.read_csv(old/'cv_grid.csv')
    registry=pd.read_csv(folder/'registry.csv')
    effects=pd.read_csv(folder/'joint_effects.csv')
    options=pd.read_csv(folder/'tied_options.csv')
    ranges=pd.read_csv(folder/'tied_ranges.csv')
    assert len(registry)==42*4 and len(effects)==63*2*2*4 and len(ranges)==63*2*4
    validated=0
    for (corpus,mode,seed),g in grid.groupby(['corpus','mode','seed']):
        def subset(frame):
            return frame[(frame.corpus==corpus)&(frame['mode']==mode)&(frame.seed==seed)]
        rank=g.sort_values('cv_hits',ascending=False,kind='stable')
        top=rank.iloc[0].cv_hits
        ties=rank[rank.cv_hits==top]
        reg=subset(registry).set_index('family').loc['joint']
        assert reg.top_cv_hits==top and reg.tied_count==len(ties)
        assert reg.runner_up_gap_hits==top-rank.iloc[1].cv_hits
        assert reg.next_distinct_gap_hits==top-rank[rank.cv_hits<top].cv_hits.max()
        for tag in (['cn'] if corpus=='cn' else ['spider','confirmation']):
            frame=pd.read_csv(old/'predictions'/f'{tag}-{mode}-{seed}.csv.gz',dtype={'id':str})
            assert frame.id.is_unique
            for stratum,q in [('all',frame),('unseen',frame[frame.stratum=='unseen'])]:
                tr=subset(ranges); tr=tr[(tr.dataset==tag)&(tr.stratum==stratum)]
                to=subset(options); to=to[(to.dataset==tag)&(to.stratum==stratum)]
                for r in tr.itertuples():
                    tg=to[to.family==r.family].sort_values('tie_order')
                    assert len(tg)==r.tied_count
                    assert (r.selected_correct,r.min_correct,r.max_correct)==(tg.iloc[0].correct,tg.correct.min(),tg.correct.max())
                    assert (tg.correct-tg.reference_correct).equals(tg.fixes-tg.harms)
                e=subset(effects); e=e[(e.dataset==tag)&(e.stratum==stratum)]
                for r in e.itertuples():
                    count_eligible=(ties.alpha==0).any()
                    count_chosen=ties.iloc[0].alpha==0 or (count_eligible and r.tie_policy=='count_first')
                    choice=q.count_cv if count_chosen else q.hybrid
                    reference=q[{'hybrid':'hybrid','fixed_count':'fixed_size','selected_count':'count_cv','full':'inverse'}[r.reference]]
                    a=choice==q.gold; b=reference==q.gold
                    assert (r.n,r.correct,r.reference_correct,r.fixes,r.harms)==(len(q),a.sum(),b.sum(),(a&~b).sum(),(~a&b).sum())
                    assert (r.alpha==0)==count_chosen
                    validated+=1
    print(f'Verified 42 grids, {len(options)} tied-option rows, {len(ranges)} ranges and {validated} joint contrasts.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('folder',type=Path)
    args=p.parse_args();run(args.root,args.folder)
