"""Report exact-set accuracies on identical candidate lists and the frozen request IDs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def log_odds(kernel, cross, targets):
    mean = kernel.mean(axis=0)
    centered = kernel - mean[:,None] - mean[None,:] + mean.mean()
    query = cross-cross.mean(axis=1,keepdims=True)-mean[None,:]+mean.mean()
    indicator = targets.mean(axis=0) + query @ np.linalg.solve(centered+np.eye(len(kernel)),targets-targets.mean(axis=0))
    low = 1/(len(kernel)+2)
    indicator = indicator.clip(low,1-low)
    return np.log(indicator/(1-indicator))

def generate(root):
    bases = pd.read_csv(root/'primary/bases/predictions.csv.gz', dtype={'id':str})
    matched = pd.read_csv(root/'extensions/matched/predictions.csv.gz',dtype={'id':str})
    result=[]
    for task,alpha,qalpha in [('cn',.25,1.),('spider',.5,2.)]:
        cohorts = {'all':None,'unseen':'unseen_combination'}
        with np.load(root/'primary/primary'/f'{task}.npz') as archive, np.load(root/'extensions/matched/inputs'/f'{task}.npz') as q:
            z={name:archive[name] for name in archive.files}
            np.testing.assert_array_equal(z['ids'],q['ids'])
            qwen = log_odds(q['kernel'],q['query_kernel'],q['targets'].astype(float))
            for model in ['dvlc','gma','adaptive','qwen']:
                same = model=='qwen'
                cache=matched[matched.task==task] if same else bases[(bases.task==task)&(bases.base==model)]
                cache=cache.set_index('id').loc[z['ids']]
                logits=qwen if same else z[model]
                predictions=[]
                for i,(a,b) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
                    masks=z['masks'][a:b]
                    v=((masks[:,None] & (1<<np.arange(logits.shape[1])))!=0).astype(float)
                    base=v@logits[i]
                    forward=base+v@z['forward_logits'][i]
                    inv=v@z['linear'][i]-.5*((v@z['gram'])*v).sum(1)
                    scores=(base,forward,forward+inv,forward+(qalpha if same else alpha)*inv)
                    predictions.append(tuple(masks[x.argmax()] for x in scores))
                pred=np.array(predictions)
                for col,chk in ([(0,'base'),(1,'forward'),(3,'inverse')] if same else [(0,'adapter'),(1,'forward'),(2,'lpr')]):
                    np.testing.assert_array_equal(pred[:,col],cache[chk])
                for cohort,label in cohorts.items():
                    which=np.ones(len(cache),dtype=bool) if label is None else cache.stratum==label
                    totals=(pred[which]==cache.loc[which,'gold'].to_numpy()[:,None]).sum(0)
                    for role,total in zip(['Base','Forward','Unit','Weighted'],totals):
                        result.append([task,model,cohort,role,int(total),int(which.sum()),100*total/which.sum()])
    return pd.DataFrame(result,columns='task model cohort scorer hits n accuracy'.split())

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('release',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    m=generate(args.release)
    m.to_csv(args.out/'base-scores.csv',index=False)
    indexed=m.set_index(['model','task','scorer','cohort']).sort_index()
    rows=[]
    for model,display in [('dvlc','DVLC'),('gma','GMA-SCL'),('adaptive','AdaptiveMLC'),('qwen','Qwen ridge')]:
        values=[]
        for task in ['cn','spider']:
            for estimator in ['Base','Forward','Unit']:
                pair=indexed.loc[(model,task,estimator)]
                a,u=(pair.loc[c,'accuracy'] for c in ['all','unseen'])
                v=f'{a:.2f}/{u:.2f}'
                if estimator=='Unit':v=r'\textbf{'+v+'}'
                values.append(v)
        rows.append(' & '.join([display]+values)+r'\\')
    latex=r"""\begin{table}[tb]\centering\footnotesize\setlength{\tabcolsep}{2.7pt}
\caption{Exact-set accuracy on all/unseen requests with a fixed unit inverse weight (\%).}\label{tab:multibase}
\begin{tabular}{@{}lrrr@{\hspace{5pt}}rrr@{}}\toprule
&\multicolumn{3}{c}{CrossWOZ}&\multicolumn{3}{c}{SpiderOps}\\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Base head & Base & +Fwd. & +Fwd.+Inv. & Base & +Fwd. & +Fwd.+Inv.\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}\end{table}
"""
    (args.out/'multibase.tex').write_text(latex,encoding='utf8')
