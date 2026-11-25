"""Rebuild frozen record posting lists in SQLite and replay the lookup evaluation."""
import argparse
import csv
import gzip
import itertools
import json
import math
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("gold", "BF", "ECP", "Row", "Spherical", "Mapping", "BF+Mapping", "Full")
STRATA = ("all", "seen", "unseen_combination", "unseen_atom")
COUNTS = ("tp", "fp", "fn", "tn", "empty_both")
LOOKUP_FIELDS = ("task", "lookup", "arity", "arm", "cohort", *COUNTS, "f1")

def queries(registry, labels):
    pairs = [p for p in itertools.combinations(range(len(labels)), 2)
             if any(mask & ((1 << p[0]) | (1 << p[1])) == ((1 << p[0]) | (1 << p[1])) for mask in registry)]
    return [(j,) for j in range(len(labels))] + pairs

def evaluate(data):
    rows = []
    returned = {}
    for task, inputs in data.items():
        labels, registry, records = (inputs[key] for key in ("labels", "registry_masks", "records"))
        ids = {r[0] for r in records}
        assert len(ids) == len(records) and len(registry) == 75
        by_cohort = dict(all=ids)
        for s in STRATA[1:]:
            by_cohort[s] = {r[0] for r in records if r[1] == s}
        assert set.union(*(by_cohort[s] for s in STRATA[1:])) == ids
        predicates = queries(registry, labels)
        assert len(predicates) - len(labels) == {"cn": 53, "spider": 47}[task]
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE posting(arm TEXT,atom INTEGER,rec TEXT,PRIMARY KEY(arm,atom,rec)) WITHOUT ROWID")
            for rec, cohort, *masks in records:
                assert len(masks) == len(ARMS)
                for arm, mask in zip(ARMS, masks):
                    assert 0 <= mask < (1 << len(labels))
                    conn.executemany("INSERT INTO posting VALUES(?,?,?)", ((arm,j,rec) for j in range(len(labels)) if mask & (1<<j)))
            for predicate in predicates:
                lookup = "+".join(labels[j] for j in predicate)
                sql = " INTERSECT ".join(["SELECT rec FROM posting WHERE arm=? AND atom=?"] * len(predicate))
                pred = {arm: {r[0] for r in conn.execute(sql, [a for j in predicate for a in (arm,j)])} for arm in ARMS}
                returned[(task, lookup)] = {arm: sorted(members) for arm, members in pred.items()}
                for arm in ARMS[1:]:
                    for cohort, universe in by_cohort.items():
                        if not universe:
                            continue
                        truth, hits = pred["gold"] & universe, pred[arm] & universe
                        tp, fp, fn = len(hits & truth), len(hits - truth), len(truth - hits)
                        empty = int(not (truth or hits))
                        rows.append(dict(task=task, lookup=lookup, arity=len(predicate), arm=arm, cohort=cohort,
                                         tp=tp, fp=fp, fn=fn, tn=len(universe)-tp-fp-fn, empty_both=empty,
                                         f1=1.0 if empty else (2*tp)/(2*tp+fp+fn)))
    return rows, returned

def aggregate(rows):
    keys = ("task", "arity", "arm", "cohort")
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    return [dict(zip(keys, key), nlookups=len(g), macro_f1=sum(r["f1"] for r in g)/len(g),
                 **{k:sum(r[k] for r in g) for k in COUNTS}) for key,g in sorted(groups.items())]

def csv_check(name, rows, fields, write):
    if write:
        with (HERE/name).open("w", encoding="utf8",newline="") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    else:
        with (HERE/name).open(encoding="utf8", newline="") as f:
            expected=list(csv.DictReader(f))
        assert len(rows)==len(expected),name
        for row, ref in zip(rows, expected):
            for k in fields:
                if isinstance(row[k],float):assert math.isclose(row[k],float(ref[k]),rel_tol=0,abs_tol=1e-12),(name,k,row,ref)
                else:assert str(row[k])==ref[k],(name,k,row,ref)

def run(write=False):
    with gzip.open(HERE/"index_inputs.json.gz","rt",encoding="utf8") as f:
        rows, returned = evaluate(json.load(f))
    summary=aggregate(rows)
    csv_check("lookups.csv",rows,LOOKUP_FIELDS,write)
    csv_check("summary.csv",summary,tuple(summary[0]),write)
    results=[dict(task=k[0],lookup=k[1],records=v) for k,v in returned.items()]
    if write:
        with gzip.open(HERE/"retrieved_ids.json.gz","wt",encoding="utf8") as f:json.dump(results,f,ensure_ascii=False,separators=(",",":"))
    else:
        with gzip.open(HERE/"retrieved_ids.json.gz","rt",encoding="utf8") as f:assert results==json.load(f)
    splits={t:{r["cohort"] for r in summary if r["task"]==t and r["cohort"]!="all"} for t in ("cn","spider")}
    for r in summary:
        if r["cohort"]=="all":
            sub=[z for z in summary if z["task"]==r["task"] and z["arm"]==r["arm"] and z["arity"]==r["arity"] and z["cohort"] in splits[r["task"]]]
            assert all(r[k]==sum(z[k] for z in sub) for k in ("tp","fp","fn","tn"))
    print(f"Verified {len(returned)} unique lookups, {len(rows)} lookup/arm/cohort rows and {len(summary)} summaries.")

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--write",action="store_true");run(p.parse_args().write)
