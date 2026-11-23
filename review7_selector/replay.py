"""Audit tied registry choices and add the zero-inverse branch to the frozen selector."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def winners(grid, hits):
    scores = sorted((hits[c] for c in grid), reverse=True)
    top = scores[0]
    next_distinct = next((s for s in scores if s < top), None)
    return [c for c in grid if hits[c] == top], top, top-scores[1], top-next_distinct if next_distinct is not None else np.nan


def performance(pred, ref, gold):
    correct, anchor = pred == gold, ref == gold
    fixes = int((correct & ~anchor).sum())
    harms = int((~correct & anchor).sum())
    return dict(correct=int(correct.sum()), reference_correct=int(anchor.sum()),
                fixes=fixes, harms=harms, change_pp=100*(fixes-harms)/len(gold))


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    old = root/'review5_selector'
    impl = load_module(old/'selector.py')
    draws = load_module(root/'round2_draws/refit_draws.py')
    cv = pd.read_csv(old/'cv_grid.csv').set_index(['corpus','mode','seed'])
    published = pd.read_csv(old/'selections.csv').set_index(['corpus','mode','seed','method'])
    registry_meta, options, effects, tied_ranges = [], [], [], []
    for corpus in ['cn','spider']:
        cohorts = ['cn'] if corpus == 'cn' else ['spider','confirmation']
        queries = {}
        for tag in cohorts:
            with np.load(root/'refit_inputs'/f'{tag}.npz') as arr:
                queries[tag] = arr['t'],arr['pq'],arr['ids'].astype(str)
        for mode,seed,ids,support in impl.iterate(root,corpus):
            key = corpus,mode,seed
            group = cv.loc[[key]]
            hits = {(r.metric,r.alpha,r.beta):int(r.cv_hits) for r in group.itertuples()}
            assert len(hits)==len(impl.GRID)==102 and len(ids)==75
            families = {'hybrid':impl.HYBRID, 'selected':impl.INVERSE, 'count_cv':impl.COUNT,
                        'joint':impl.GRID}
            choices = {}
            meta = dict(corpus=corpus, mode=mode, seed=seed)
            for name,grid in families.items():
                win,top,gap,distinct_gap=winners(grid,hits)
                winner=win[0]
                if name in ['hybrid','selected','count_cv']:
                    record=published.loc[key+(name,)]
                    assert (record.metric,record.alpha,record.beta)==winner
                    assert int(record.ties)==len(win) and record.cv_hits==top
                registry_meta.append(dict(**meta,family=name,top_cv_hits=top,
                    winner_metric=winner[0],winner_alpha=winner[1],winner_beta=winner[2],
                    tied_count=len(win), runner_up_gap_hits=gap,
                    next_distinct_gap_hits=distinct_gap,
                    tied_metrics='|'.join(dict.fromkeys(c[0] for c in win))))
                choices[name]=win
            opposite = next((c for c in choices['joint'] if c[1]==0), choices['joint'][0])
            eval_configs = list(dict.fromkeys(c for win in choices.values() for c in win))
            for tag,(t,pq,query_ids) in queries.items():
                prediction=impl.score(draws,support,(t,pq),eval_configs)
                pred={c:prediction[:,i] for i,c in enumerate(eval_configs)}
                ref=pd.read_csv(old/'predictions'/f'{tag}-{mode}-{seed}.csv.gz',dtype={'id':str})
                assert np.array_equal(query_ids,ref.id)
                for family in ('selected','count_cv','hybrid'):
                    assert np.array_equal(pred[choices[family][0]],ref[family])
                for part,mask in [('all',np.ones(len(ref),bool)),('unseen',ref.stratum.eq('unseen').to_numpy())]:
                    gold=ref.gold.to_numpy()[mask]
                    full=ref.inverse.to_numpy()[mask]
                    baseline=ref.hybrid.to_numpy()[mask]
                    eval_meta=dict(**meta,dataset=tag,stratum=part,n=len(gold))
                    for name,win in choices.items():
                        observed=[]
                        for i,c in enumerate(win):
                            y=pred[c][mask]
                            outcome=performance(y,full,gold)
                            options.append(dict(**eval_meta, family=name, tie_order=i+1,
                                metric=c[0],alpha=c[1],beta=c[2],cv_hits=hits[c],**outcome))
                            observed.append(outcome['correct'])
                        tied_ranges.append(dict(**eval_meta,family=name,selected_correct=observed[0],
                            full_correct=int((full==gold).sum()),tied_count=len(win),
                            min_correct=min(observed),max_correct=max(observed),
                            distinct_correct_counts=len(set(observed)),
                            distinct_prediction_vectors=len({pred[c][mask].tobytes() for c in win})))
                    for priority,winner in [('inverse_first',choices['joint'][0]),('count_first',opposite)]:
                        for reference,anchor in [('hybrid',baseline),('full',full),('fixed_count',ref.fixed_size.to_numpy()[mask]),('selected_count',ref.count_cv.to_numpy()[mask])]:
                            effects.append(dict(**eval_meta,tie_policy=priority,reference=reference,
                                selected_metric=winner[0],alpha=winner[1],beta=winner[2],
                                **performance(pred[winner][mask],anchor,gold)))
    for name,table in [('registry',registry_meta),('tied_options',options),
                       ('tied_ranges',tied_ranges),('joint_effects',effects)]:
        pd.DataFrame(table).to_csv(out/f'{name}.csv',index=False,float_format='%.10g')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        run(args.archive,args.output)
