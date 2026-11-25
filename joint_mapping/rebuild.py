"""Reconstruct unscaled candidate summands from the frozen replay."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def pack(task, ref, mapper):
    md = ref / 'mapping'
    with np.load(md / 'inputs' / (task + '.npz')) as z:
        a, y, soft = (z[k].astype(float) for k in ('a','y','soft'))
        ids, reg = z['ids'], z['reg']
    cv = pd.read_csv(md/'oof.csv.gz', dtype={'id':str})
    key = cv.loc[cv.task.eq(task)].drop_duplicates('id').set_index('id').loc[reg]
    bits = 1 << np.arange(y.shape[1])
    np.testing.assert_array_equal(y.astype(int)@bits,key.gold)
    o = pd.read_csv(ref/'decoder'/'results'/'loss_inputs.csv.gz',dtype={'id':str})
    held = {k:d for k,d in o[o.task.eq(task)&o.arm.eq('full')].groupby('id',sort=False)}
    rows = {}
    for fold in range(5):
        h = key.fold.eq(fold).to_numpy()
        assert set(key.group[h]).isdisjoint(key.group[~h])
        observed_a, observed_b = a[~h].any(0), y[~h].any(0)
        m = mapper(a[~h][:,observed_a],y[~h][:,observed_b],soft[:len(reg)][h][:,observed_a])
        for i, k in enumerate(reg[h]):
            d = held[k]
            masks = d.candidate.to_numpy()
            v = ((masks[:,None] & bits[observed_b]) != 0).astype(float)
            np.testing.assert_array_equal(v.astype(int)@bits[observed_b], masks)
            rows[k] = np.column_stack([masks,d.pre,v@m[i],d.inverse])
    # Registration rows precede all evaluation rows to preclude label use in tuning.
    records = [rows[k] for k in reg]
    m = mapper(a,y,soft[len(reg):])
    with np.load(ref/'primary'/(task+'.npz')) as r:
        np.testing.assert_array_equal(ids,r['ids'])
        masks,linear,gram,offsets=(r[k] for k in ('masks','linear','gram','offsets'))
        un = r['forward_logits']+r['dvlc'].astype(float)
        for i,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
            opts = masks[start:stop]
            v = ((opts[:,None]&bits)!=0).astype(float)
            inv = v@linear[i]-.5*((v@gram)*v).sum(1)
            records.append(np.column_stack([opts,v@un[i],v@m[i],inv]))
    arr=np.vstack(records)
    pred = pd.read_csv(md/'predictions.csv.gz',dtype={'id':str})
    pred=pred.loc[pred.task.eq(task)].set_index('id').loc[ids]
    np.savez_compressed(HERE/(task+'.npz'), reg=reg, ids=ids, groups=pred.group.astype(str).to_numpy(dtype=str), fold=key.fold.to_numpy(),
        rgold=key.gold.to_numpy(), egold=pred.gold.to_numpy(), unseen=pred.stratum.eq('unseen_combination').to_numpy(),
        offs=np.r_[0,np.cumsum([len(row) for row in records])], masks=arr[:,0].astype(np.int32),
        J=arr[:,1], M=arr[:,2], I=arr[:,3], old_sequential=pred.map_inverse.to_numpy(),
        old_primary=pred.primary.to_numpy(), old_mapping_only=pred.map_only.to_numpy())


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('numeric_inputs_replay',type=Path)
    ref=ap.parse_args().numeric_inputs_replay
    spec=importlib.util.spec_from_file_location('mapping',ref/'mapping'/'refit_mapping.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for t in ['cn','spider']:
        pack(t,ref,mod.mapper)

