"""Index diagnostic. Fix workload to every registered atom and every pair
observed together in the registry. No evaluation annotations select lookups.
Compare Base+Forward vs primary LPR and soft mapping. A false posting is one
wrong query-record membership. Macro-F1 gives each lookup equal weight; 1 for
a query with empty gold and predicted retrieval sets. All requests participate.
"""
import sys, sqlite3,itertools
from pathlib import Path
import numpy as np
import pandas as pd
def run(compact,dest):
    data=pd.read_csv(compact/'interface/predictions.csv.gz',dtype={'id':str})
    missing=pd.read_csv(compact/'cohort/unsupported_targets.csv',dtype={'id':str}).set_index(['task','id'])
    rows=[]
    for task,frame in data.groupby('task'):
        z=np.load(compact/'primary'/f'{task}.npz'); labels=list(z['labels'])
        y=np.load(compact/'mapping/inputs'/f'{task}.npz')['y']
        queries=[(i,) for i in range(len(labels))]
        queries += [(i,j) for i,j in itertools.combinations(range(len(labels)),2) if np.logical_and(y[:,i],y[:,j]).any()]
        gold=frame.gold.to_numpy().copy()
        for i in np.where(gold<0)[0]:
            atomset=missing.loc[(task,frame.id.iat[i]),'labels'].split('|')
            gold[i]=sum(1<<j for j,lab in enumerate(labels) if lab in atomset)
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE posting (arm TEXT, atom INTEGER, rec INTEGER, PRIMARY KEY(arm,atom,rec)) WITHOUT ROWID')
            for arm,masks in [('gold',gold),*[(k,frame[k]) for k in ('base_forward','full_selected','mapping')]]:
                db.executemany('INSERT INTO posting VALUES(?,?,?)',((arm,j,i) for i,mask in enumerate(masks) for j in range(len(labels)) if mask&(1<<j)))
            for q in queries:
                expr=' INTERSECT '.join(['SELECT rec FROM posting WHERE arm=? AND atom=?']*len(q))
                def fetch(arm):return set(r[0] for r in db.execute(expr,list(itertools.chain.from_iterable((arm,j) for j in q))))
                truth=fetch('gold')
                for arm in ('base_forward','full_selected','mapping'):
                    pred=fetch(arm)
                    tp=len(truth&pred); fp=len(pred-truth); fn=len(truth-pred)
                    rows.append((task,'+'.join(labels[i] for i in q),len(q),arm,tp,fp,fn,2*tp/(2*tp+fp+fn) if tp+fp+fn else 1))
    out=pd.DataFrame(rows,columns='task lookup arity arm tp fp fn f1'.split())
    out.to_csv(dest/'lookups.csv',index=False)
    print(out.groupby(['task','arity','arm']).agg(lookups=('f1','count'),macro_f1=('f1','mean'),false_postings=('fp','sum'),missed_postings=('fn','sum')).to_string())
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))

