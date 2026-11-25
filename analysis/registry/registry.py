"""Summarize and draw registry-level paired changes from frozen exact-set predictions."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

WEIGHTS = (1, .5, 2, .25, 0)
TRANSFERRED = {'cn': .25, 'spider': .5}
DATASETS = {'cn': 'CrossWOZ', 'spider': 'SpiderOps'}
METHODS = ['Forward', 'ECP', 'Row', 'LPR, unit weight', 'LPR, transferred weight', 'LPR, registry-selected']
COHORTS = ['All', 'Draw-unseen']


def collect(folder):
    rows = []
    for key, name in DATASETS.items():
        with np.load(folder/f'{key}.npz', allow_pickle=False) as data:
            assert len(data['mode']) == 20
            for i, (policy, seed) in enumerate(zip(data['mode'], data['seed'])):
                observed = data['cv_oof'][i] == data['cv_gold'][i, :, None, None]
                chosen = observed.sum(axis=0).argmax(axis=1)
                np.testing.assert_array_equal(chosen, data['cv_selected'][i])
                predictions = data['query_pred'][i]
                gold = data['query_gold'][i]
                pred = {
                    'Forward': predictions[:, 0, WEIGHTS.index(0)],
                    'ECP': predictions[:, 2, chosen[2]],
                    'Row': predictions[:, 1, chosen[1]],
                    'LPR, unit weight': predictions[:, 0, WEIGHTS.index(1)],
                    'LPR, transferred weight': predictions[:, 0, WEIGHTS.index(TRANSFERRED[key])],
                    'LPR, registry-selected': predictions[:, 0, chosen[0]],
                }
                selected = pred['LPR, registry-selected'] == gold
                for cohort, mask in zip(COHORTS, (np.ones(len(gold), bool), data['query_unseen'][i])):
                    for method in METHODS:
                        correct = (pred[method] == gold)
                        fixes = int(np.sum(mask & selected & ~correct))
                        harms = int(np.sum(mask & ~selected & correct))
                        n = int(mask.sum())
                        hits = int((mask & correct).sum())
                        rows.append((name, str(policy), int(seed), cohort, method, WEIGHTS[chosen[0]],
                                     n, hits, 100*hits/n, fixes, harms, 100*(fixes-harms)/n))
    return pd.DataFrame(rows, columns=('task policy seed cohort method selected_weight n hits accuracy '
                                       'fixes harms gain').split())


def table(detail, out):
    means = detail.groupby(['method', 'task', 'cohort']).accuracy.mean().unstack(['task','cohort'])
    cols = [(d, c) for d in DATASETS.values() for c in COHORTS]
    means = means.loc[METHODS, cols]
    rows = [r'\begin{table}[!htbp]\centering\small',
            r'\caption{Mean standalone exact-set accuracy (\%) across 20 registries per task.}\label{tab:reselection}',
            r'\begin{tabular}{@{}lrrrr@{}}\toprule',
            r'&\multicolumn{2}{c}{CrossWOZ}&\multicolumn{2}{c}{SpiderOps}\\',
            r'\cmidrule(lr){2-3}\cmidrule(l){4-5} Scorer & All & Draw-unseen & All & Draw-unseen\\\midrule']
    for method in METHODS:
        values = [format(v, '.2f') if v < means[c].max() else
                  (r'\textbf{' + format(v, '.2f')+'}') for c,v in means.loc[method].items()]
        rows.append(method + ' & ' + ' & '.join(values) + r'\\')
    rows.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (out/'registry-table.tex').write_text('\n'.join(rows)+'\n', encoding='utf8')
    means.to_csv(out/'registry-means.csv')
    return means


def draw(detail, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.spines.right':False,'axes.spines.top':False})
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 1.87), sharex=True, sharey=True)
    fig.subplots_adjust(left=.175, right=.985, bottom=.22, top=.78, wspace=.14)
    bands = [('All','Forward'),('All','ECP'),('Draw-unseen','Forward'),('Draw-unseen','ECP')]
    color = {'Forward':'#156884', 'ECP':'#9a4d18'}
    marker = {'Forward':'o', 'ECP':'^'}
    for ax, task in zip(axes, DATASETS.values()):
        subset = detail[detail.task==task]
        ax.axvline(0, color='#434343', linewidth=.65, zorder=1)
        ax.axhline(1.5, color='#c0c0c0', linewidth=.6, zorder=1)
        ax.grid(axis='x', color='#e1e1e1', linewidth=.5)
        for index,(cohort,method) in enumerate(bands):
            records=subset[(subset.cohort==cohort)&(subset.method==method)].sort_values(['policy','seed'])
            assert len(records)==20
            values=records.gain.to_numpy()
            # Stack nearby values vertically to keep coincident observations visible.
            used=[]
            for value in sorted(values):
                offsets=[0,.078,-.078,.156,-.156,.234,-.234,.312,-.312]
                level=min(offsets,key=lambda h:sum(abs(v-value)<1.36 and abs(s-h)<.073 for v,s in used))
                used.append((value,level))
            ax.scatter([p[0] for p in used],[3-index+p[1] for p in used],s=11, marker=marker[method],
                       facecolors='none',edgecolors=color[method],linewidths=.6,zorder=3)
        ax.set_title(task,fontsize=9,fontweight='semibold', pad=2)
        ax.set_xlim(-5,36)
        ax.set_xticks([0,10,20,30])
        ax.set_ylim(-.5,3.5)
        ax.set_yticks([3,2,1,0],['All / Fwd','All / ECP','Unseen / Fwd','Unseen / ECP'])
        ax.tick_params(axis='both',length=0,pad=2,labelsize=7.3)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#777777')
    largest=detail[detail.method.isin(['Forward','ECP'])].gain
    assert largest.min()>-5 and largest.max()<36, (largest.min(), largest.max())
    fig.supxlabel('Selected LPR minus comparator (accuracy points)',fontsize=8,y=.022)
    fig.savefig(out/'registry-gains.pdf',bbox_inches='tight',pad_inches=.022,
                metadata={'Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(out/'registry-gains.png',dpi=190,bbox_inches='tight',pad_inches=.022)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='directory containing cn.npz and spider.npz from the standalone registry grids')
    parser.add_argument('output',type=Path)
    parser.add_argument('--no-plot',action='store_true',help='only recompute the numerical tables with NumPy and pandas')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    detail=collect(args.input)
    assert len(detail)==480
    detail.to_csv(args.output/'registry-draws.csv',index=False)
    print(table(detail,args.output).round(2).to_string())
    if not args.no_plot:
        draw(detail,args.output)
