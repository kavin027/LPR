"""Score fixed registered posting-list predicates and replay paired group intervals."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ('Forward', 'ECP', 'Row', 'Spherical', 'LPR', 'Unit', 'BaseInv', 'Mapping')
METRICS = ('macro_f1', 'micro_f1', 'precision', 'recall')
REPLICATES = 2000
SEED = 251121


def data(path):
    frame = pd.read_csv(path, dtype={'id': str, 'group': str})
    return frame.set_index(['task', 'id'], verify_integrity=True)


def load(root, task):
    raw = root / 'primary'
    registry = np.load(raw / f'mapping/inputs/{task}.npz')['y'].astype(bool)
    with np.load(raw / f'primary/{task}.npz') as source:
        labels = list(source['labels'].astype(str))
    assert registry.shape == (75, len(labels)) and registry.any(axis=0).all()
    ref = data(raw / 'interface/predictions.csv.gz').loc[task]
    gold = ref.gold.copy()
    for key, names in data(raw / 'cohort/unsupported_targets.csv').labels.items():
        t, rid = key
        if t == task:
            assert gold.loc[rid] < 0
            gold.loc[rid] = sum(1 << i for i, name in enumerate(labels) if name in names.split('|'))
    assert (gold >= 0).all()
    cal = data(raw / 'cal/predictions.csv.gz').loc[task].loc[ref.index]
    metric = data(raw / 'metric/results/predictions.csv.gz').loc[task].loc[ref.index]
    all_rules = pd.read_csv(raw / 'decoder/results/predictions.csv.gz', dtype={'id': str})
    inverse = all_rules[(all_rules.task == task) & (all_rules.rule == 'base_inverse')].set_index('id', verify_integrity=True)
    masks = dict(Forward=ref.base_forward, ECP=cal.ecp_selected, Row=cal.row_selected, Spherical=metric.isotropic_selected,
                 LPR=ref.full_selected, Unit=cal.full_fixed, BaseInv=inverse.loc[ref.index, 'pred'], Mapping=ref.mapping)
    return labels, registry, ref.group, gold.to_numpy(), masks


def fscore(tp, predicted, relevant):
    denom = predicted + relevant
    return np.divide(2.*tp, denom, out=np.ones(denom.shape, dtype=float), where=denom != 0)


def group_totals(values, codes, ngroup):
    result = np.zeros((ngroup, values.shape[1]), dtype=np.int32)
    np.add.at(result, codes, values)
    return result


def estimate(root):
    totals, contrasts, detail, postings = [], [], [], []
    for task in ('cn', 'spider'):
        labels, registry, group, gold, masks = load(root, task)
        groups = sorted(set(group), key=int if task == 'cn' else str)
        codes = pd.Categorical(group, categories=groups).codes
        # A posting is one unique (atom, record ID) entry.
        old_postings = masks['Forward'].astype(int).to_list()
        new_postings = masks['LPR'].astype(int).to_list()
        insertions = sum((new & ~old).bit_count() for old, new in zip(old_postings, new_postings))
        deletions = sum((old & ~new).bit_count() for old, new in zip(old_postings, new_postings))
        old_count = sum(mask.bit_count() for mask in old_postings)
        new_count = sum(mask.bit_count() for mask in new_postings)
        assert old_count + insertions - deletions == new_count
        postings.append(dict(task=task, records=len(gold), base_forward_rows=old_count, lpr_rows=new_count,
                             insertions=insertions, deletions=deletions,
                             records_changed=sum(a != b for a,b in zip(old_postings,new_postings))))
        n = len(groups)
        weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=REPLICATES).astype(float)
        for arity in (1, 2):
            members = [inds for inds in combinations(range(len(labels)), arity)
                       if registry[:, list(inds)].all(axis=1).any()]
            predicates = np.array([sum(1 << i for i in inds) for inds in members])
            relevant = (gold[:, None] & predicates) == predicates
            g = group_totals(relevant, codes, n)
            gboot = weights @ g
            points, boot, f1s = {}, {}, {}
            for arm in ARMS:
                found = (masks[arm].to_numpy()[:, None] & predicates) == predicates
                p = group_totals(found, codes, n)
                t = group_totals(found & relevant, codes, n)
                tt, pp, gg = t.sum(axis=0), p.sum(axis=0), g.sum(axis=0)
                f1s[arm] = fscore(tt, pp, gg)
                points[arm] = 100*f1s[arm].mean()
                boot[arm] = 100*fscore(weights @ t, weights @ p, gboot).mean(axis=1)
                tp, pred, truth = tt.sum(), pp.sum(), gg.sum()
                totals.append(dict(task=task, arity=arity, arm=arm, predicates=len(members), groups=n,
                                   tp=tp, fp=pred-tp, fn=truth-tp, macro_f1=points[arm], micro_f1=200*tp/(pred+truth),
                                   precision=100*tp/pred, recall=100*tp/truth))
                for ix, idxs in enumerate(members):
                    detail.append(dict(task=task, arity=arity, arm=arm, predicate='+'.join(labels[i] for i in idxs),
                                       tp=tt[ix], fp=pp[ix]-tt[ix], fn=gg[ix]-tt[ix], f1=100*f1s[arm][ix]))
            for other in ('Forward', 'Spherical'):
                delta = f1s['LPR'] - f1s[other]
                lo, hi = np.percentile(boot['LPR'] - boot[other], [2.5, 97.5])
                contrasts.append(dict(task=task, arity=arity, reference=other, gain=points['LPR']-points[other],
                                      low=lo, high=hi, wins=(delta>0).sum(), ties=(delta==0).sum(), losses=(delta<0).sum()))
    return {'metrics': pd.DataFrame(totals), 'paired': pd.DataFrame(contrasts), 'lookups': pd.DataFrame(detail), 'postings': pd.DataFrame(postings)}


def verify(root, output, write=False):
    result = estimate(root)
    for name, frame in result.items():
        target = output / f'{name}.csv'
        if write:
            frame.to_csv(target, index=False)
        else:
            actual = pd.read_csv(target)
            pd.testing.assert_frame_equal(frame, actual, check_exact=False, rtol=1e-10, atol=1e-10)
    print(result['paired'].round(3).to_string(index=False))
    print('PASS: registered single-label and label-pair retrieval, including paired intervals.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path, help='release directory containing primary/')
    parser.add_argument('--write', action='store_true', help='regenerate reference CSVs')
    args = parser.parse_args()
    verify(args.release, Path(__file__).resolve().parent, args.write)
