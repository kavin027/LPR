"""Report fixed-request gaps and their change from the paired 75-record anchor."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from freeze import OUT

def main():
    curve=pd.read_csv(OUT/'curve.csv')
    df=pd.read_csv(OUT/'predictions.csv.gz',dtype={'id':str,'group':str})
    for meth in ['direct','forward','inverse','direct_cv']:
        df[meth+'_ok']=df[meth].eq(df.gold).astype(int)
    records=[]
    for ds,q in df.groupby('dataset',sort=False):
        for stratum,part in [('all',q),('fixed_unseen',q[q.orig_unseen])]:
            anchor=part[part.budget==75].set_index('id')
            for lhs,rhs in [('inverse','direct'),('forward','direct'),('inverse','forward'),('inverse','direct_cv')]:
                left,right=lhs+'_ok', rhs+'_ok'
                for budget,z in part.groupby('budget'):
                    per=z.groupby('id')[[left,right]].mean().loc[anchor.index]
                    gap=per[left]-per[right]
                    diff=gap-(anchor[left]-anchor[right])
                    codes,groups=pd.factorize(anchor.group,sort=True)
                    weights=np.random.default_rng(251121).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
                    boot=100*(weights@np.bincount(codes,weights=diff))/(weights@np.bincount(codes))
                    records.append(dict(dataset=ds,stratum=stratum,budget=budget,method=lhs,reference=rhs,delta_gap_pp=100*diff.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975)))
    pd.DataFrame(records).to_csv(OUT/'gap_change_vs_75.csv',index=False)
    names={'cn':'CrossWOZ','spider':'Spider to BREAK'}
    colors={'direct':'#3868a3','forward':'#d07a23','inverse':'#19806b'}
    labels={'direct':'Train B directly','forward':'Forward with legacy A','inverse':'Forward + inverse'}
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.0),layout='constrained')
    for row,part in enumerate(['all','fixed_unseen']):
        for col,ds in enumerate(names):
            ax=axs[row,col]
            here=curve[(curve.stratum==part)&(curve.dataset==ds)]
            for k in colors:ax.plot(here.budget,here[k],marker='o',ms=4,lw=2,color=colors[k],label=labels[k])
            ax.set_xscale('log',base=2)
            ticks=[75,150,300,600,1200,here.budget.max()]
            ax.set_xticks(ticks,[str(n) for n in ticks],fontsize=8)
            ax.set_ylim(0,100)
            ax.grid(axis='y',alpha=.22)
            ax.set_title(names[ds]+' | '+('all test requests' if row==0 else 'unseen at 75, fixed requests'),fontsize=11)
            if col==0:ax.set_ylabel('Exact-set accuracy (%)')
            if row==1:ax.set_xlabel('B-labeled training requests')
    handles,texts=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,texts,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(OUT/'learning_curves.png',dpi=160)
    plt.close(fig)
if __name__=='__main__': main()
