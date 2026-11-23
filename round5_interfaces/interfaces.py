"""Recompute a full-inverse/fixed-count comparison at three legacy API resolutions."""
import argparse
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expose(scores, mode):
    if mode == 'dense':
        return scores
    if mode == 'decisions':
        return (scores >= 0.5).astype(float)
    keep = np.argsort(-scores, axis=1, kind='stable')[:, :3]
    out = np.zeros_like(scores)
    np.put_along_axis(out, keep, 1, axis=1)
    return out


def evaluate(root, directory):
    draws = module(root/'round2_draws/refit_draws.py')
    mechanism = module(root/'round2_mechanism/compute.py')
    summary = []
    directory.mkdir(parents=True, exist_ok=True)
    for tag in ('cn','spider','confirmation'):
        with np.load(root/'refit_inputs'/f'{tag}.npz') as f:
            s,t,p,pq,a,y,g,ids = [f[k] for k in ('s','t','p','pq','a','y','groups','ids')]
        bits = 1 << np.arange(y.shape[1])
        prior = np.log1p(np.bincount(y.sum(1).astype(int), minlength=len(bits)+1))
        reference = pd.read_csv(root/'controls'/f'{tag}_requests.csv', dtype={'id':str})
        assert np.array_equal(reference.id, ids)
        for mode in ('dense','top3','decisions'):
            r,q = expose(p,mode),expose(pq,mode)
            _,_,pr = draws.forward(s,t,r,q,a,y,g)
            gram,linear = mechanism.geom(y,r,q,g,'full')
            logits = np.log(pr/(1-pr))
            masks = list(draws.options(pr,y))
            results = []
            for i,opts in enumerate(masks):
                v = ((opts[:,None]&bits)!=0).astype(float)
                u = v@logits[i]
                size = opts[(u+prior[v.sum(1).astype(int)]).argmax()]
                full = opts[(u+v@linear[i]-0.5*np.sum((v@gram)*v,axis=1)).argmax()]
                results.append((ids[i],size,full))
            df = pd.DataFrame(results,columns=['id','fixed_size','inverse'])
            if mode=='dense':
                for arm in ('fixed_size','inverse'):
                    assert np.array_equal(df[arm],reference[arm])
            df['gold'] = reference.gold
            df['stratum'] = reference.stratum
            df['reachable'] = [t in o for t,o in zip(reference.gold,masks)]
            df.to_csv(directory/f'{tag}-{mode}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
            for name,part in [('all',df),('unseen',df[df.stratum=='unseen_combination'])]:
                h,b = part.inverse==part.gold,part.fixed_size==part.gold
                summary.append([tag,mode,name,len(part),int(part.reachable.sum()),int(h.sum()),int(b.sum()),int((h&~b).sum()),int((b&~h).sum())])
    frame=pd.DataFrame(summary,columns=['dataset','interface','stratum','n','reachable','full_hits','prior_hits','fixes','harms'])
    frame.to_csv(directory/'summary.csv',index=False)
    print(frame[frame.stratum=='unseen'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        evaluate(args.archive_root,args.output)

