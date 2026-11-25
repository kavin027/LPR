from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent
s=pd.read_csv(p/'summary.csv'); t=pd.read_csv(p/'perturbations.csv')
def stats(task,mode,cohort,ref):
    z=s[(s.kind=='standalone')&(s.task==task)&(s.design==mode)&(s.cohort==cohort)&(s.reference==ref)].iloc[0]
    return f"{z['mean']:+.2f} & {z.sd:.2f} & {z.worst:+.2f} & {z.wins}/{z.ties}/{z.losses}"
T={'cn':'CrossWOZ','spider':'SpiderOps'}
intro=r"""\section*{Paired gains across registry draws}
For every draw we recompute the exact-hit totals on held-out registration groups for each scorer and apply its selected coefficient to the same evaluation requests. Each draw contributes equally. The standard deviation uses the 20 paired accuracy differences, and the minimum is the least favorable of those differences. The draws share records and test requests. Their dispersion describes these workloads.

\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}
\caption{Standalone Full relative to each reference across 20 registries per task.}
\begin{tabular}{@{}lllrrrr@{}}\toprule
Task & Cohort & Reference & Mean & SD & Min & W/T/L\\\midrule
"""
rows=[]
for task in T:
 for c in ('All','Draw-unseen'):
  for ref in ('ECP','Row','Forward'):
   rows.append(f"{T[task]} & {c} & {ref} & "+stats(task,'pooled',c,ref)+r"\\")
 rows.append(r'\midrule')
tail=r"""
\bottomrule\end{tabular}\end{table}

On each task, 19 of 20 standalone selections assign Full a positive coefficient. Those 19 outperform Forward on both cohorts. In the remaining CrossWOZ and SpiderOps draws, respectively, dropping ECP gains 58 and loses 2 draw-unseen exact matches, and gains 4 and loses 1. The corresponding Full and Forward predictions coincide on every request. In the separate DVLC series the first CrossWOZ replacement registry also selects zero, with 12 fixes and 2 harms against ECP on its unseen requests and zero changes against Base + Forward. The neural sensitivity runs comprise three seeds on the original registry and two replacement registries per task. They use 40 additional neural training folds in total.

\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}
\caption{Fresh minus transferred Full by registry sampling policy (10 draws per cell).}
\begin{tabular}{@{}lllrrrr@{}}\toprule
Task & Registry & Cohort & Mean & SD & Min & W/T/L\\\midrule
"""
b=[]
for task in T:
 for mode,label in [('enriched','Capped'),('uniform','Uncapped')]:
  for c in ('All','Draw-unseen'):
   b.append(f"{T[task]} & {label} & {c} & "+stats(task,mode,c,'Transferred')+r"\\")
 b.append(r'\midrule')
end=r"""
\bottomrule\end{tabular}\end{table}

\paragraph{Sensitivity to the choice of draws and query groups.}
Each of 2,000 paired perturbations resamples ten draw indices with replacement within each policy and resamples whole query dialogues or databases, using the same group multiplicities for every draw. We first compute the accuracy difference on each resampled cohort and then average the resampled draws equally. This holds the training pool, fitted predictions and two sampling policies fixed. The middle 95\% of perturbations for fresh minus transferred Full spans $[-2.55,-0.36]$ points overall and $[-2.91,0.47]$ on draw-unseen CrossWOZ. The SpiderOps ranges are $[-1.53,-0.31]$ and $[-1.15,0.09]$. Thus the lower observed mean under fresh selection is sensitive to which draws enter the unseen comparison. These ranges leave uncertainty about a new training population or a new business migration unmeasured.
"""
(p/'registry88.tex').write_text(intro+'\n'.join(rows[:-1])+tail+'\n'.join(b[:-1])+end,encoding='utf8')
