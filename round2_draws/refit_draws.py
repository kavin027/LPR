"""Replay the prespecified 75-row strong-comparator repeats from the anonymous artifact."""
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve, cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
GRID = np.array([0, .25, .5, 1, 2, 4.])


def read(p):
    return json.loads(p.read_bytes())


def calibration(x, y, groups, query):
    xc = x - x.mean(0)
    gram = xc.T @ xc + np.eye(x.shape[1])
    hat = xc @ solve(gram, xc.T, assume_a="pos") + 1/len(x)
    resid = y - hat @ y
    held = np.empty_like(y)
    for g in np.unique(groups):
        ids = np.flatnonzero(groups == g)
        held[ids] = y[ids] - solve(np.eye(len(ids)) - hat[np.ix_(ids, ids)], resid[ids], assume_a="pos")
    slope = solve(gram, xc.T @ (y-y.mean(0)), assume_a="pos")
    return held, y.mean(0)+(query-x.mean(0))@slope


def forward(s,t,p,pq,a,y,g):
    cr,cq=calibration(p,a,g,pq)
    f=np.column_stack([s,p,cr.clip(0,1)])
    z=np.column_stack([t,pq,cq.clip(0,1)])
    x=f-f.mean(0)
    kernel=x@x.T
    kernel.flat[::len(y)+1]+=1
    prob=(z-f.mean(0))@x.T@cho_solve(cho_factor(kernel),y-y.mean(0))+y.mean(0)
    lo=1/(len(y)+2)
    return f,z,prob.clip(lo,1-lo)


def options(prob,y):
    bits=1<<np.arange(y.shape[1])
    known=np.unique(y.astype(int)@bits)
    m=min(5,len(bits))
    flips=(np.arange(1<<m)[:,None] & (1<<np.arange(m)))!=0
    first=(prob>=.5)@bits
    near=np.argsort(abs(prob-.5),axis=1,kind='stable')[:,:m]
    for b,ix in zip(first,near):
        choices=np.union1d(known,b^(flips@bits[ix]))
        yield np.r_[b, choices[choices!=b]]


def cv_weight(s,p,a,y,g):
    wins=np.zeros(len(GRID),int)
    for group in np.unique(g):
        held=g==group
        use=y[~held].any(0); use_old=a[~held].any(0)
        yr=y[~held][:,use]
        _,_,pr=forward(s[~held],s[held],p[~held],p[held],a[~held][:,use_old],yr,g[~held])
        bits=1<<np.arange(yr.shape[1])
        counts=np.log1p(np.bincount(yr.sum(1).astype(int),minlength=len(bits)+1))
        truth=y[held][:,use].astype(int)@bits
        eligible=~y[held][:,~use].any(1)
        for i,opts in enumerate(options(pr,yr)):
            v=((opts[:,None]&bits)!=0).astype(float)
            u=v@np.log(pr[i]/(1-pr[i]))
            pred=opts[np.argmax(u[:,None]+counts[v.sum(1).astype(int),None]*GRID,axis=0)]
            wins+=(pred==truth[i])&eligible[i]
    return float(GRID[wins.argmax()])


def source(y,p,g):
    def fit(take):
        x=y[take]; z=p[take]
        weight=1/np.maximum(1,x.sum(1))
        mx,mz=[np.average(v,weights=weight,axis=0) for v in (x,z)]
        xw=(x-mx).T*weight
        b=np.linalg.solve(xw@(x-mx)+np.eye(x.shape[1]), xw@(z-mz))
        return mz-mx@b,b
    residual=np.empty_like(p)
    for group in np.unique(g):
        ix=g==group
        a,b=fit(~ix)
        residual[ix]=p[ix]-a-y[ix]@b
    var=max(1e-12,np.mean(residual**2/np.maximum(1,y.sum(1,keepdims=True))))
    return *fit(np.ones(len(y),bool)),var


def arff(path,x,y,L):
    with path.open('w',encoding='ascii') as out:
        out.write('@relation transfer\n')
        for j in range(L):out.write(f'@attribute l{j} {{0,1}}\n')
        for j in range(x.shape[1]):out.write(f'@attribute x{j} numeric\n')
        out.write('@data\n')
        for i,row in enumerate(x):
            head=','.join(['?']*L if y is None else [str(int(v)) for v in y[i]])
            out.write(head+','+','.join(format(v,'.17g') for v in row)+'\n')


def br(f,z,y,opts,work,classpath):
    _,sv,vh=np.linalg.svd(f,full_matrices=False)
    basis=vh[sv>sv[0]*max(f.shape)*np.finfo(float).eps].T
    arff(work/'support.arff',f@basis,y,y.shape[1]);arff(work/'query.arff',z@basis,None,y.shape[1])
    (work/'options.txt').write_text(''.join(' '.join(map(str,v))+'\n' for v in opts),encoding='ascii')
    subprocess.run(['java','-cp',classpath,'IndependentSMO',str(work),str(y.shape[1])],check=True,
                   stdout=subprocess.DEVNULL)
    return np.loadtxt(work/'br.txt',dtype=int)


def predict(tag,s,t,p,pq,a,y,g,work,classpath):
    f,z,pr=forward(s,t,p,pq,a,y,g)
    opts=list(options(pr,y))
    bits=1<<np.arange(y.shape[1])
    hist=np.bincount(y.sum(1).astype(int),minlength=len(bits)+1)
    yc=y-y.mean(0); pc=p-p.mean(0)
    b=np.linalg.solve(yc.T@yc+np.eye(y.shape[1]),yc.T@pc)
    res,_=calibration(y,p,g,y[:0])
    cov=LedoitWolf().fit(p-res).covariance_
    ev,vec=np.linalg.eigh(cov)
    m=vec/np.sqrt(np.maximum(ev,max(1e-12,1e-8*ev.max())))
    w=b@m; obs=(pq-p.mean(0)+y.mean(0)@b)@m
    coef=cv_weight(s,p,a,y,g) if tag=='cn' else None
    if tag!='cn':
        intercept,source_b,sigma=source(y,p,g)
        iso=max(float(cov.diagonal().mean()),1e-12)
    choices=[]
    for i,c in enumerate(opts):
        v=((c[:,None]&bits)!=0).astype(float); k=v.sum(1).astype(int)
        u=v@np.log(pr[i]/(1-pr[i]))
        row={'inverse':u-.5*((v@w-obs[i])**2).sum(1),'fixed_size':u+np.log1p(hist[k])}
        if tag=='cn':row['global_cv']=u+coef*np.log1p(hist[k])
        else:
            variance=sigma*np.maximum(1,k)
            row['forward_convolution']=u-.5*(((v@source_b+intercept-pq[i])**2).sum(1)/variance+p.shape[1]*np.log(variance))
            row['isotropic']=u-.5*((v@b+p.mean(0)-y.mean(0)@b-pq[i])**2).sum(1)/iso
        choices.append({key:int(c[np.argmax(val)]) for key,val in row.items()})
    df=pd.DataFrame(choices)
    if tag!='cn':df['independent_smo']=br(f,z,y,opts,work,classpath)
    return df,opts,coef


def evaluate_frame(df,opts,rows,labels,ys):
    bits={name:1<<i for i,name in enumerate(labels)}
    known=np.unique(ys.astype(int)@(1<<np.arange(len(labels))))
    gold=np.array([sum(bits[v] for v in row['new']) if all(v in bits for v in row['new']) else -1 for row in rows])
    df=df.copy()
    df['id']=[row['id'] for row in rows]
    df['gold']=gold
    df['stratum']=np.where(gold<0,'unsupported',np.where(np.isin(gold,known),'seen','unseen'))
    df['reachable']=[target in c for target,c in zip(gold,opts)]
    return df


def run(root,meka,output,tags):
    reg=root/'registration';ctrl=root/'controls';mech=root/'mechanism'
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(4):
        work=Path(tmp)
        libs=str(meka.resolve()/'lib'/'*');cp=str(work)+os.pathsep+libs
        if any(k!='cn' for k in tags):
            subprocess.run(['javac','-cp',libs,'-d',str(work),str(ctrl/'MatchedEnsemble.java'),str(HERE/'IndependentSMO.java')],check=True)
        for tag in tags:
            # Check original 75 registrations using identical frozen numerical inputs.
            with np.load(root/'refit_inputs'/f'{tag}.npz') as inp:
                pred,_,_=predict(tag,*[inp[k] for k in ['s','t','p','pq','a','y','groups']],work,cp)
            ref=pd.read_csv(ctrl/f'{tag}_blind.csv')
            for col in ['inverse','fixed_size','global_cv' if tag=='cn' else 'independent_smo']:
                assert np.array_equal(pred[col],ref[col]),(tag,col,np.sum(pred[col]!=ref[col]))
            if tag!='cn':
                other=pd.read_csv(mech/('spider_predictions.csv' if tag=='spider' else 'new_database_secondary_predictions.csv'))
                for col in ['forward_convolution','isotropic']:
                    assert np.array_equal(pred[col],other[col]),(tag,col)
            print('Original registry identity OK:',tag,len(pred),flush=True)
            corpus='cn' if tag=='cn' else 'spider'
            rows=read(reg/f'{corpus}_rows.json')
            index={v['id']:i for i,v in enumerate(rows['register'])}
            draws=[r for r in read(reg/f'{corpus}_selections.json') if r['budget']==75]
            panel=np.load(reg/f'{corpus}_panel.npz')
            s,p,t,pq=[panel[k] for k in ['s','p','t','pq']]
            queries=rows['queries']
            if tag=='confirmation':
                with np.load(root/'refit_inputs'/'confirmation.npz') as inp:
                    t,pq,ids=[inp[k] for k in ['t','pq','ids']]
                truth=read(reg/'confirmation_gold.json')
                queries=[dict(id=i,new=truth[i]) for i in ids]
            refm=np.load(reg/f'{corpus}_predictions.npz')
            refsum=pd.read_csv(reg/'confirmation_sensitivity.csv') if tag=='confirmation' else None
            for r in draws:
                ix=[index[v] for v in r['ids']]
                rs=[rows['register'][i] for i in ix]; labels=r['labels']
                oldlabels=sorted({a for v in rs for a in v['old']})
                ind=lambda name,vs:np.array([[v in row[name] for v in vs] for row in rs],float)
                y=ind('new',labels);a=ind('old',oldlabels);g=np.array([row['group'] for row in rs])
                out,opts,coef=predict(tag,s[ix],t,p[ix],pq,a,y,g,work,cp)
                if tag!='confirmation':
                    stem=f"{r['seed']}-{r['mode']}-75-"
                    assert np.array_equal(refm[stem+'1'],out.inverse) and np.array_equal(refm[stem+'size'],out.fixed_size)
                frame=evaluate_frame(out,opts,queries,labels,y)
                if tag=='confirmation':
                    original=refsum[(refsum['mode']==r['mode'])&(refsum.seed==r['seed'])&(refsum.budget==75)&(refsum.alpha==1)]
                    for rec in original.itertuples():
                        keep=np.ones(len(frame),bool) if rec.stratum=='all' else frame.stratum==rec.stratum
                        assert rec.inverse_hits==((frame.gold==frame.inverse)&keep).sum()
                        assert rec.size_hits==((frame.gold==frame.fixed_size)&keep).sum()
                frame['mode']=r['mode'];frame['seed']=r['seed'];frame['weight']=coef
                frame['registered_atoms']=len(labels)
                frame['registered_sets']=len(np.unique(y.astype(int)@(1<<np.arange(len(labels)))))
                prefix=f"{tag}-{r['mode']}-{r['seed']}"
                frame.to_csv(output/f'{prefix}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                print(prefix,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_root',type=Path);parser.add_argument('meka_root',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('tags',nargs='*',default=['cn','spider','confirmation'])
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    run(args.artifact_root,args.meka_root,args.output,args.tags)

