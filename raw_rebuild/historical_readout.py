"""Use rebuilt group-ordered embeddings and old labels to refit the legacy ridge observation model."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('vectors',type=Path,help='encoder.py output ordered as history, register, query')
    ap.add_argument('history_jsonl',type=Path)
    ap.add_argument('output',type=Path)
    ap.add_argument('--compare',type=Path,help='optional refit_inputs/cn.npz or spider.npz from the artifact')
    args=ap.parse_args()
    examples=[json.loads(x) for x in args.history_jsonl.read_text(encoding='utf8').splitlines()]
    n=len(examples)
    vocab=sorted({atom for r in examples for atom in r['old']})
    y=np.array([[int(t in r['old']) for t in vocab] for r in examples],dtype=float)
    with np.load(args.vectors) as cache:
        ids=cache['ids'];vectors=cache['vectors'].astype(float)
    assert list(ids[:n])==[r['id'] for r in examples]
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    x,q=vectors[:n],vectors[n:];mu=x.mean(0);xc=x-mu; offset=y.mean(0)
    w=solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y-offset),assume_a='pos')
    scores=(offset+(q-mu)@w).clip(0,1)
    np.savez_compressed(args.output,ids=ids[n:],old_scores=scores,old_labels=vocab)
    if args.compare:
        with np.load(args.compare) as want:
            assert list(ids[n:])==list(want['register_ids'])+list(want['ids'])
            diff=np.abs(scores-np.vstack((want['p'],want['pq'])))
            print('max absolute difference',diff.max(),'mean',diff.mean())
if __name__=='__main__':
    with threadpool_limits(2):main()
