"""Audit selector and population contrasts from an extracted anonymous release."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SEED = 251121
B = 2000
REFERENCES = ('inverse', 'fixed_size', 'count_cv')


def aligned(path, ids):
    frame = pd.read_csv(path, dtype={'id': str}).set_index('id', verify_integrity=True)
    assert len(frame) == len(ids) and frame.index.difference(ids).empty
    return frame.loc[ids]


def integers(group_n, rng):
    return rng.multinomial(group_n, np.full(group_n, 1/group_n), size=B)


def original_comparisons(tag, frame):
    rows = []
    for st, sub in [('all', frame), ('unseen', frame.loc[frame.stratum.eq('unseen')])]:
        # Match the fixed-registry audit's ordering for finite Monte Carlo replicates.
        labels = list(sub.group.unique())
        labels.sort(key=int if all(v.isdigit() for v in labels) else str)
        g = pd.Categorical(sub.group, categories=labels).codes
        weights = integers(len(labels), np.random.default_rng(SEED))
        denominator = weights @ np.bincount(g, minlength=len(labels))
        h = sub.hybrid.eq(sub.gold).to_numpy()
        for reference in REFERENCES:
            r = sub[reference].eq(sub.gold).to_numpy()
            d = h.astype(int) - r.astype(int)
            ci = np.quantile(100*(weights @ np.bincount(g, weights=d, minlength=len(labels))) / denominator,
                             [.025, .975])
            rows.append(dict(dataset=tag, population=st, method='hybrid', reference=reference,
                             n=len(sub), groups=len(labels), hits=h.sum(), reference_hits=r.sum(),
                             fixes=(d==1).sum(), harms=(d==-1).sum(), delta_pp=100*d.mean(),
                             lower_pp=ci[0], upper_pp=ci[1], replicates=B))
    return rows


def crossed_selector(tag, mode, frames, wc, wr, group_ids):
    rows = []
    group_n = wc.shape[1]
    for st in ('all','unseen'):
        mask = np.stack([np.ones(len(f),bool) if st=='all' else f.stratum.eq('unseen').to_numpy() for f in frames])
        sizes = np.stack([np.bincount(group_ids, weights=k, minlength=group_n) for k in mask])
        totals = wc @ sizes.T
        good = ((totals > 0)|(wr==0)).all(axis=1)
        for reference in REFERENCES:
            differences = []
            for f,m in zip(frames,mask,strict=True):
                delta = f.hybrid.eq(f.gold).astype(int) - f[reference].eq(f.gold).astype(int)
                differences.append(np.bincount(group_ids, weights=m*delta.to_numpy(), minlength=group_n))
            differences = np.stack(differences)
            per_draw = 100*differences.sum(1)/sizes.sum(1)
            ratios = np.divide(100*(wc @ differences.T), totals, out=np.zeros_like(totals), where=totals>0)
            boot = (wr*ratios).sum(1)[good]/len(frames)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(dataset=tag,mode=mode,population=st,method='hybrid',reference=reference,
                             draws=len(frames),query_groups=group_n,mean_n=sizes.sum(1).mean(),
                             wins=(per_draw>0).sum(),ties=(per_draw==0).sum(),losses=(per_draw<0).sum(),
                             delta_pp=per_draw.mean(),draw_sd_pp=per_draw.std(ddof=1),
                             lower_pp=lo,upper_pp=hi,valid=len(boot)))
    return rows


def cross_membership(tag, mode, original, redraws, seeds):
    rows=[]
    for seed,draw in zip(seeds,redraws,strict=True):
        for fit,frame in [('original',original),('redraw',draw)]:
            # Use the population-defining fit only for membership. Masks and correctness
            # are compared inside the scoring fit's own local vocabulary.
            for population,keep in [('original unseen',original.stratum.eq('unseen')),
                                    ('redraw unseen',draw.stratum.eq('unseen')),
                                    ('all',np.ones(len(frame),bool))]:
                part=frame.loc[keep]
                full=part.inverse.eq(part.gold)
                prior=part.fixed_size.eq(part.gold)
                rows.append(dict(dataset=tag,mode=mode,seed=seed,fit=fit,population=population,n=len(part),
                                 full_hits=full.sum(),prior_hits=prior.sum(),
                                 fixes=(full&~prior).sum(),harms=(~full&prior).sum(),
                                 full_acc=100*full.mean(),prior_acc=100*prior.mean(),
                                 delta_pp=100*(full.mean()-prior.mean())))
    return rows


def population_breadth(tag, original):
    q=original[original.stratum=='unseen'].copy()
    q['full_hit']=q.inverse.eq(q.gold)
    q['prior_hit']=q.fixed_size.eq(q.gold)
    q['row_fix']=q.full_hit&~q.prior_hit
    q['row_harm']=~q.full_hit&q.prior_hit
    records=[]
    for dimension in ('gold','group'):
        grouped=q.groupby(dimension).agg(n=('gold','size'),full_hits=('full_hit','sum'),prior_hits=('prior_hit','sum'),
                                          fixes=('row_fix','sum'),harms=('row_harm','sum')).reset_index()
        grouped['net_hits']=grouped.fixes-grouped.harms
        grouped['dataset']=tag
        grouped['dimension']=dimension
        records.append(grouped.rename(columns={dimension:'value'}))
    sets,groups=records
    desc=dict(dataset=tag,requests=len(q),sets=len(sets),groups=len(groups),most_frequent_set_n=sets.n.max(),
              full_hits=q.full_hit.sum(),prior_hits=q.prior_hit.sum(),fixes=q.row_fix.sum(),harms=q.row_harm.sum(),
              request_acc=100*q.full_hit.mean(),set_macro_acc=100*(sets.full_hits/sets.n).mean(),
              total_net=(q.row_fix.sum()-q.row_harm.sum()),best_set_net=sets.net_hits.max(),
              best_two_groups_net=groups.nlargest(2,'net_hits').net_hits.sum(),
              group_positive=(groups.net_hits>0).sum(),group_zero=(groups.net_hits==0).sum(),group_negative=(groups.net_hits<0).sum())
    return records,desc


def numerical_sensitivity(root,out):
    metrics=pd.read_csv(root/'review5_registry/registry_metrics.csv')
    m=metrics.query("tag=='cn' and mode=='uniform' and budget==25 and stratum=='all'").copy()
    assert len(m)==10 and m.full_equivalent_ties.sum()==292
    m['replayed_net_hits']=m.full_hits-m.archived_full_hits
    m['archived_against_diag_pp']=100*(m.archived_full_hits-m.diagonal_hits)/m.n
    m['replayed_against_diag_pp']=100*(m.full_hits-m.diagonal_hits)/m.n
    m[['seed','n','full_equivalent_ties','archived_full_hits','full_hits','diagonal_hits','replayed_net_hits',
       'archived_against_diag_pp','replayed_against_diag_pp']].to_csv(out/'small_budget_ties.csv',index=False)
    paths=root/'review6_encoder'
    a=pd.read_csv(paths/'archived-requests.csv.gz',dtype={'id':str}).set_index('id',verify_integrity=True)
    b=aligned(paths/'qwen_plain-requests.csv.gz',a.index)
    assert a.gold.equals(b.gold) and a.stratum.equals(b.stratum)
    rows=[]
    for st,m in [('all',np.ones(len(a),bool)),('unseen',a.stratum.eq('unseen_combination'))]:
        for arm in ('inverse','fixed_size'):
            original=a[arm].loc[m]; current=b[arm].loc[m]; gold=a.gold.loc[m]
            old=original.eq(gold); new=current.eq(gold)
            rows.append(dict(population=st,method=arm,n=len(gold),changed_masks=(original!=current).sum(),
                             archived_hits=old.sum(),plain_hits=new.sum(),fixes=(~old&new).sum(),
                             harms=(old&~new).sum(),delta_pp=100*(new.mean()-old.mean())))
    frame=pd.DataFrame(rows)
    check=frame[(frame.population=='unseen')&(frame.method=='inverse')].iloc[0]
    assert check.n==1223 and check.changed_masks==13 and check.plain_hits==611
    frame.to_csv(out/'fresh_encoder.csv',index=False)


def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    original_rows=[]; crossed_rows=[]; cells=[]; dimensions=[]; breadth=[]
    for tag in ('cn','spider','confirmation'):
        base=pd.read_csv(root/'controls'/f'{tag}_requests.csv',dtype={'id':str,'group':str}).set_index('id',verify_integrity=True)
        folder=root/'review5_selector/predictions'
        original=aligned(folder/f'{tag}-original-0.csv.gz',base.index)
        assert all(original[c].equals(base[c]) for c in ('gold','inverse','fixed_size'))
        original['group']=base.group
        original_rows.extend(original_comparisons(tag,original))
        detail,desc=population_breadth(tag,original)
        dimensions.extend(detail);breadth.append(desc)
        group_ids=pd.factorize(base.group,sort=True)[0]
        group_n=int(group_ids.max())+1
        rng=np.random.default_rng(SEED)
        wc=integers(group_n,rng)
        wr=integers(10,rng)
        for mode in ('uniform','enriched'):
            paths=sorted(folder.glob(f'{tag}-{mode}-*.csv.gz'))
            seeds=[int(p.name.removesuffix('.csv.gz').rsplit('-',1)[1]) for p in paths]
            assert set(seeds)==set(range(1,11))
            frames=[aligned(p,base.index) for p in paths]
            crossed_rows.extend(crossed_selector(tag,mode,frames,wc,wr,group_ids))
            cells.extend(cross_membership(tag,mode,original,frames,seeds))
    pd.DataFrame(original_rows).to_csv(out/'original_selector.csv',index=False)
    pd.DataFrame(crossed_rows).to_csv(out/'crossed_selector.csv',index=False)
    pd.DataFrame(breadth).to_csv(out/'breadth.csv',index=False)
    pd.concat(dimensions,ignore_index=True).to_csv(out/'breadth_by_group_or_set.csv',index=False)
    c=pd.DataFrame(cells)
    c.to_csv(out/'per_draw_crossed_population.csv',index=False)
    c.groupby(['dataset','mode','fit','population'])[['n','full_hits','prior_hits','fixes','harms','full_acc','prior_acc','delta_pp']].mean().reset_index().to_csv(out/'crossed_population.csv',index=False)
    numerical_sensitivity(root,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path,help='extracted anonymous release')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):run(args.root,args.output)
