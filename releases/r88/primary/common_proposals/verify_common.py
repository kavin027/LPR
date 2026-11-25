"""Recompute the matched-proposal accuracy table and its Base + Inverse ablation."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ['task', 'base', 'id']
BASES = ('dvlc', 'gma', 'adaptive')

def load(relative):
    return pd.read_csv(ROOT / relative, dtype={'id': str, 'group': str})

def verify():
    base = load('bases/predictions.csv.gz')
    extra = load('common_proposals/additional_predictions.csv.gz')
    rank = load('rank5/predictions.csv.gz')
    records = base.merge(extra, on=KEY, validate='one_to_one')
    records = records.merge(rank[KEY + ['rank5']], on=KEY, validate='one_to_one')
    assert len(records) == len(rank) == len(extra) == len(base) == 14133
    expected = load('common_proposals/reported_accuracy.csv')
    assert len(expected) == 90 and not expected.duplicated(['task', 'base', 'method']).any()
    for task, bases in records.groupby('task'):
        for name, predictions in bases.groupby('base'):
            unseen = predictions[predictions.stratum == 'unseen_combination']
            assert len(unseen) == {'cn':1223, 'spider':548}[task]
            methods = expected[(expected.task == task) & (expected.base == name)]
            assert len(methods) == 15
            for method in methods.itertuples():
                observed = 100 * unseen[method.method].eq(unseen.gold).mean()
                assert abs(observed - method.unseen_accuracy) < 0.00500001, (task, name, method.method)
        with np.load(ROOT/'primary'/f'{task}.npz') as scores:
            gram = scores['gram']
            bits = 1 << np.arange(len(gram))
            offsets = scores['offsets']
            masks = scores['masks']
            projected = scores['linear']
            ids = scores['ids']
            terms = {b: scores[b].astype(float) for b in BASES}
            actual = {b: [] for b in BASES}
            for i,(start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                candidates = masks[start:stop]
                members = ((candidates[:, None] & bits) != 0).astype(float)
                penalty_part = .5 * ((members @ gram) * members).sum(axis=1)
                inverse = members @ projected[i] - penalty_part
                inv_mask = candidates[inverse.argmax()]
                for b in BASES:
                    base_inverse = members @ terms[b][i] + inverse
                    actual[b].append([inv_mask, candidates[base_inverse.argmax()]])
            for b in BASES:
                reference = bases[bases.base == b].set_index('id').loc[ids]
                np.testing.assert_array_equal(actual[b], reference[['inv', 'base_inv']])
    print('PASS: all 90 common-proposal accuracies and 14,133 inverse-only / Base + Inverse prediction pairs.')

if __name__=='__main__':
    verify()

