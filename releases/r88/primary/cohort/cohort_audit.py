"""Descriptive audits from archived predictions, with equal weight per registry draw."""
import argparse
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

TASKS = {'cn': 'CrossWOZ', 'spider': 'SpiderOps'}

def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)

def write_csv(path, records):
    with open(path, 'w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

def per_draw(replay):
    counts = defaultdict(Counter)
    seen = defaultdict(set)
    for r in rows(replay/'redraw/predictions.csv.gz'):
        if r['base'] != 'standalone':
            continue
        key = (r['task'], r['mode'], int(r['seed']))
        assert r['id'] not in seen[key]
        seen[key].add(r['id'])
        o, d = (r[k]=='True' for k in ('original_unseen','draw_unseen'))
        if not (o or d):
            continue
        group = 'both' if o and d else 'original_only' if o else 'draw_only'
        tally = counts[key + (group,)]
        tally['n'] += 1
        tally['unsupported_in_draw'] += int(r['gold']) < 0
        tally['full'] += int(r['full']==r['gold'])
        tally['row'] += int(r['row']==r['gold'])
        tally['fix'] += int(r['full']==r['gold'] and r['row']!=r['gold'])
        tally['harm'] += int(r['full']!=r['gold'] and r['row']==r['gold'])
    draw_rows = []
    for (t,m,s,g), v in sorted(counts.items()):
        draw_rows.append(dict(task=t,design=m,seed=s,cohort=g,n=v['n'],
                             unsupported_in_draw=v['unsupported_in_draw'],fixes=v['fix'],harms=v['harm'],full_hits=v['full'],row_hits=v['row'],
                             gain_pp=100*(v['full']-v['row'])/v['n']))
    summary = []
    for task in TASKS:
        for cohort in ('both','original_only','draw_only'):
            parts=[v for v in draw_rows if v['task']==task and v['cohort']==cohort]
            assert len(parts)==20 and len(set((p['design'],p['seed']) for p in parts))==20
            summary.append(dict(task=task,cohort=cohort,draws=len(parts),
                                mean_n=mean(p['n'] for p in parts),
                                mean_unsupported_in_draw=mean(p['unsupported_in_draw'] for p in parts),
                                mean_gain_pp=mean(p['gain_pp'] for p in parts)))
    return draw_rows, summary

def sizes(part, add_labels):
    tally=defaultdict(Counter)
    for r in part:
        mask=int(r['gold'])
        if mask < 0:
            markers=add_labels[(r['task'],r['id'])].split('|')
            key=('unsupported',tuple(sorted(markers)))
            k=len(markers)
        else:
            key=('supported',mask)
            k=mask.bit_count()
        v=tally[key]
        v['k']=k
        v['n']+=1
        f=int(r['full_selected'])==mask
        a=int(r['row_selected'])==mask
        v['full']+=f
        v['row']+=a
        v['fix']+=f and not a
        v['harm']+=a and not f
    return tally

def summarize(part):
    vals=list(part)
    n=sum(v['n'] for v in vals)
    f=sum(v['full'] for v in vals)
    r=sum(v['row'] for v in vals)
    fm=100*mean(v['full']/v['n'] for v in vals)
    rm=100*mean(v['row']/v['n'] for v in vals)
    return dict(requests=n,sets=len(vals),full_hits=f,row_hits=r,
                fixes=sum(v['fix'] for v in vals),harms=sum(v['harm'] for v in vals),
                full_acc=100*f/n,row_acc=100*r/n,gain_pp=100*(f-r)/n,
                full_macro=fm,row_macro=rm,macro_gain_pp=fm-rm)

def calibrated(replay, unsupported_file):
    add_labels={(r['task'],r['id']):r['labels'] for r in rows(unsupported_file)}
    raw={(r['task'],r['id']):r for r in rows(replay/'primary/row_predictions.csv.gz') if r['base']=='dvlc'}
    assert {k for k,r in raw.items() if int(r['gold'])<0}==set(add_labels)
    combined=[]
    for r in rows(replay/'cal/predictions.csv.gz'):
        actual=raw[(r['task'],r['id'])]
        assert actual['row']==r['row_fixed'] and actual['full']==r['full_fixed']
        combined.append({**actual,**r})
    assert len(combined)==len(raw)==len({(r['task'],r['id']) for r in combined})
    out=[]
    for task in TASKS:
        for cohort in ('all','unseen'):
            part=[r for r in combined if r['task']==task and (cohort=='all' or r['stratum']=='unseen_combination')]
            tally=sizes(part,add_labels)
            whole=list(tally.values())
            groups=[('overall','all',whole)]
            for k in sorted({v['k'] for v in whole}):
                groups.append(('cardinality',str(k),[v for v in whole if v['k']==k]))
            for name,lo,hi in [('singleton',1,1),('2-9',2,9),('10+',10,len(part))]:
                vals=[v for v in whole if lo<=v['n']<=hi]
                if vals:
                    groups.append(('frequency',name,vals))
            for axis,g,vs in groups:
                out.append(dict(task=task,cohort=cohort,axis=axis,group=g,**summarize(vs)))
    return out

def run(replay,output):
    output.mkdir(exist_ok=True,parents=True)
    draws,summary=per_draw(replay)
    write_csv(output/'overlap_draws.csv',draws)
    write_csv(output/'overlap_summary.csv',summary)
    w=calibrated(replay,Path(__file__).with_name('unsupported_targets.csv'))
    write_csv(output/'target_summary.csv',w)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path,help='Root of the extracted numerical replay')
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.replay,a.output)
