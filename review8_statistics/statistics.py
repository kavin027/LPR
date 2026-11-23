"""Cross registry fits and unseen populations on the frozen, ID-aligned query pool."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


SEED, B = 251121, 2000
STATES = ('seen', 'unseen', 'unsupported')
CELLS = ('OO', 'RO', 'OR', 'RR')  # fit first, membership second
CONTRASTS = (('RO', 'OO'), ('RR', 'OO'), ('RR', 'OR'), ('OR', 'OO'))


def read_fit(path, ids):
    f = pd.read_csv(path, dtype={'id': str}).set_index('id', verify_integrity=True)
    assert len(f) == len(ids) and ids.difference(f.index).empty
    f = f.loc[ids]
    state = f.stratum.replace({'unseen_atom': 'unsupported'}).to_numpy()
    assert set(state) <= set(STATES)
    hits = np.vstack([f.inverse == f.gold, f.fixed_size == f.gold])
    assert not hits[:, state == 'unsupported'].any()
    return state, hits.astype(int)


def multiplicities(count, rng):
    return rng.multinomial(count, np.full(count, 1 / count), size=B)


def run(root, out):
    out.mkdir(exist_ok=True, parents=True)
    cell_rows, detail_rows, contrasts, transitions = [], [], [], []
    for dataset in ('cn', 'spider', 'confirmation'):
        base = pd.read_csv(root/'controls'/f'{dataset}_requests.csv', dtype={'id': str, 'group': str})
        ids = pd.Index(base.id)
        assert ids.is_unique and base.group.notna().all()
        group, labels = pd.factorize(base.group, sort=True)
        folder = root/'review5_selector/predictions'
        origin_state, origin_hits = read_fit(folder/f'{dataset}-original-0.csv.gz', ids)
        rng = np.random.default_rng(SEED)
        wc = multiplicities(len(labels), rng)
        wr = multiplicities(10, rng)
        for mode in ('uniform', 'enriched'):
            paths = sorted(folder.glob(f'{dataset}-{mode}-*.csv.gz'))
            seeds = [int(p.name.removesuffix('.csv.gz').rsplit('-', 1)[1]) for p in paths]
            assert sorted(seeds) == list(range(1, 11))
            bootstrap = {cell: [] for cell in CELLS}
            points = {cell: [] for cell in CELLS}
            valid = {cell: [] for cell in CELLS}
            for seed, path in zip(seeds, paths, strict=True):
                draw_state, draw_hits = read_fit(path, ids)
                for old, new in product(STATES, repeat=2):
                    transitions.append(dict(dataset=dataset, mode=mode, draw=seed,
                                       original_status=old, redraw_status=new,
                                       requests=int(((origin_state == old) & (draw_state == new)).sum())))
                for cell in CELLS:
                    hits = origin_hits if cell[0] == 'O' else draw_hits
                    state = origin_state if cell[1] == 'O' else draw_state
                    keep = state == 'unseen'
                    full, count = hits[:, keep]
                    n = int(keep.sum())
                    assert n > 0
                    diff = full - count
                    delta = 100 * diff.sum() / n
                    h = dict(dataset=dataset, mode=mode, draw=seed, cell=cell,
                             n=n, full_hits=full.sum(), count_hits=count.sum(),
                             fixes=(diff == 1).sum(), harms=(diff == -1).sum(),
                             full_acc=100 * full.sum()/n, count_acc=100 * count.sum()/n, delta_pp=delta)
                    cell_rows.append(h)
                    size = np.bincount(group[keep], minlength=len(labels))
                    net = np.bincount(group[keep], weights=diff, minlength=len(labels))
                    denom = wc @ size
                    valid[cell].append(denom > 0)
                    bootstrap[cell].append(np.divide(100*(wc @ net), denom, where=denom > 0,
                                                     out=np.zeros(B)))
                    points[cell].append(delta)
            for a, b in CONTRASTS:
                # Estimate both paired contrasts on the identical resampled query groups
                # and registry draws, recalculating each membership-specific denominator.
                paired = np.stack(bootstrap[a], axis=1) - np.stack(bootstrap[b], axis=1)
                ok = ((np.stack(valid[a], axis=1) & np.stack(valid[b], axis=1)) | (wr == 0)).all(axis=1)
                changes = np.array(points[a]) - np.array(points[b])
                for seed, change in zip(seeds, changes, strict=True):
                    detail_rows.append(dict(dataset=dataset, mode=mode, draw=seed,
                                            contrast=f'{a}-{b}', delta_pp=change))
                boot = (wr * paired).sum(axis=1)[ok] / len(paths)
                lo, hi = np.quantile(boot, [.025, .975])
                contrasts.append(dict(dataset=dataset, mode=mode, contrast=f'{a}-{b}',
                                      delta_pp=changes.mean(), positive=(changes > 0).sum(),
                                      ties=(changes == 0).sum(), negative=(changes < 0).sum(),
                                      lower_pp=lo, upper_pp=hi, valid=len(boot)))
    draws = pd.DataFrame(cell_rows)
    summary = draws.groupby(['dataset','mode','cell'])[
        ['n','full_hits','count_hits','fixes','harms','full_acc','count_acc','delta_pp']
    ].mean().reset_index()
    # Independent agreement with the previous release, including both denominators and arms.
    previous = pd.read_csv(root/'review7_statistics/crossed_population.csv')
    for row in summary.itertuples(index=False):
        fit = 'original' if row.cell[0]=='O' else 'redraw'
        pop = ('original' if row.cell[1]=='O' else 'redraw') + ' unseen'
        ref = previous[(previous.dataset==row.dataset)&(previous['mode']==row.mode)&
                       (previous.fit==fit)&(previous.population==pop)].iloc[0]
        np.testing.assert_allclose([row.n, row.full_acc, row.count_acc, row.delta_pp],
                                   [ref.n, ref.full_acc, ref.prior_acc, ref.delta_pp], rtol=1e-12)
    summary.to_csv(out/'cells.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv', index=False)
    pd.DataFrame(detail_rows).to_csv(out/'per_draw.csv', index=False)
    pd.DataFrame(transitions).to_csv(out/'transitions.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.root, args.output)
