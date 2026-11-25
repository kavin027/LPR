"""Check public, parsed language-model predictions against the paired summaries."""
from pathlib import Path
import numpy as np
import pandas as pd

here=Path(__file__).resolve().parent/'results'
data=pd.read_csv(here/'predictions.csv.gz',dtype={'id':str,'group':str})
assert not data.duplicated(['task','id']).any()
for row in pd.read_csv(here/'summary.csv').itertuples():
    d=data[data.task==row.task]
    if row.cohort=='unseen':d=d[d.stratum=='unseen_combination']
    if row.cohort=='seen':d=d[d.stratum=='seen']
    prediction=d.pred.eq(d.gold)&~d.invalid
    reference=d[row.comparator].eq(d.gold)&(d.gold>=0)
    fix=prediction & ~reference
    harm=reference & ~prediction
    assert (len(d),sum(prediction),sum(reference),sum(d.invalid),sum(d.outside),sum(d.outside & prediction),sum(fix),sum(harm))==(row.n,row.hits,row.ref_hits,row.malformed,row.list_external,row.correct_external,row.fixes,row.harms)
    groups=sorted(d.group.unique(),key=int if row.task=='cn' else str)
    ix=pd.Categorical(d.group,categories=groups).codes
    weights=np.random.default_rng(251121).multinomial(len(groups),np.full(len(groups),1/len(groups)),2000)
    paired=100*(weights@np.bincount(ix,weights=fix.astype(int)-harm.astype(int)))/(weights@np.bincount(ix))
    np.testing.assert_allclose([100*prediction.mean(),100*reference.mean(),100*(fix.sum()-harm.sum())/len(d),*np.percentile(paired,[2.5,97.5])],[row.accuracy,row.ref_accuracy,row.gain,row.low,row.high],atol=1e-9)
print('PASS: open generation exact-set scores, format failures and grouped intervals from parsed masks.')
