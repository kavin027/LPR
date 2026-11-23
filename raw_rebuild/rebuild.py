"""Reconstruct role-ordered requests from upstream training files and public IDs."""
import argparse
import ast
import csv
import hashlib
import io
import json
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp

HERE=Path(__file__).resolve().parent
SQL=set("select where join group having order limit distinct union intersect except count sum avg min max and or not eq neq gt gte lt lte in like between exists add sub mul div".split())
OPS=set("select project filter aggregate group superlative comparative union intersection discard sort boolean arithmetic comparison".split())

def load(path):
    return json.loads(path.read_bytes())

def norm(text):
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())

def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def crosswoz(raw):
    revised=defaultdict(set)
    appearances=0
    with zipfile.ZipFile(raw/"data.zip") as z:
        for name in z.namelist():
            if "/crosswoz/" not in name or not name.endswith("/train.json"):continue
            for domain,panels in json.loads(z.read(name)).items():
                for panel in panels:
                    for episode in panel:
                        for role in episode.values():
                            for chars,labels in zip(role["seq_ins"],role["labels"],strict=True):
                                revised["".join(chars)].add((domain,tuple(sorted(set(labels)))))
                                appearances+=1
    originals=defaultdict(list)
    rows={}
    with zipfile.ZipFile(raw/"crosswoz_train.zip") as z:
        for g,conv in json.loads(z.read("train.json")).items():
            for i,v in enumerate(conv["messages"]):
                if v["role"]!="usr": continue
                uid=f"{g}:{i}"
                olds=sorted({json.dumps(x[:3],ensure_ascii=False,separators=(",",":")) for x in v["dialog_act"]})
                rows[uid]=dict(id=uid,group=g,text=norm(v["content"]),old=olds)
                originals[v["content"]].append(uid)
    counts=Counter(appearances=appearances,distinct_revised=len(revised),upstream_user_turns=len(rows))
    for text,assign in revised.items():
        matches=originals.get(text,[])
        if not matches:counts["unmatched_revised"]+=1;continue
        if len(matches)>1:counts["ambiguous_original"]+=1;continue
        if len(assign)>1:counts["ambiguous_revised"]+=1;continue
        rows[matches[0]]["new"]=list(next(iter(assign))[1])
        counts["unique_alignments"]+=1
    return rows,counts

def spider(raw):
    old=pd.read_parquet(raw/"spider.train.parquet")
    data=pd.read_csv(raw/"break.train.csv")
    data=data[(data.split=="train") & data.question_id.str.startswith("SPIDER_train_")]
    ids=Counter(data.question_id)
    rows=[]
    counts=Counter(spider_train=len(old),break_spider=len(data))
    for b in data.itertuples():
        i=int(b.question_id.rsplit("_",1)[1])
        if ids[b.question_id]!=1 or i>=len(old) or b.question_text!=old.iloc[i].question:
            counts["unaligned"]+=1;continue
        try:
            t=sqlglot.parse_one(old.iloc[i].query,read="sqlite")
            atoms={n.key for n in t.walk()} & SQL
            if len(list(t.find_all(exp.Select)))>1:atoms.add("multiple_select")
            if not atoms:raise ValueError("no SQL atom")
        except (ValueError,sqlglot.errors.ParseError):
            counts["sql_failure"]+=1;continue
        try:
            ops=ast.literal_eval(b.operators)
            if not isinstance(ops,list) or not ops or not set(ops)<=OPS:raise ValueError("operations")
        except (SyntaxError,ValueError,TypeError):
            counts["operation_failure"]+=1;continue
        rows.append(dict(id=b.question_id,group=old.iloc[i].db_id,text=norm(b.question_text),old=sorted(atoms),new=sorted(set(ops))))
    freq=Counter(v["text"] for v in rows)
    counts["duplicate_text_rows"]=sum(n for n in freq.values() if n>1)
    output={v["id"]:v for v in rows if freq[v["text"]]==1}
    counts["eligible"]=len(output)
    return output,counts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path,help="directory with four upstream files named as in sources.json")
    p.add_argument("output",type=Path,help="local output directory")
    p.add_argument("--evidence",type=Path,help="paired-evidence.zip, or a zip containing paired_predictions.csv")
    opt=p.parse_args()
    for k,m in load(HERE/"sources.json")["files"].items():
        if sha(opt.raw/k)!=m["sha256"]:raise ValueError("source digest mismatch: "+k)
    roles=load(HERE/"roles.json")
    truth={}
    if opt.evidence:
        with zipfile.ZipFile(opt.evidence) as z:
            key=next(k for k in z.namelist() if k.split("/")[-1]=="paired_predictions.csv")
            rr=list(csv.DictReader(io.StringIO(z.read(key).decode("utf-8-sig"))))
            truth={(r["dataset"],r["request"]):(r["group"],sorted(json.loads(r["truth"]))) for r in rr}
            assert len(truth)==len(rr)==4711
    out=opt.output;out.mkdir(exist_ok=True,parents=True)
    summary={}
    for task,builder in (("cn",crosswoz),("spider",spider)):
        rows,counts=builder(opt.raw)
        summary[task]=dict(counts)
        groups=set()
        for role,ids in roles[task].items():
            assert len(ids)==len(set(ids))
            gs={rows[k]["group"] for k in ids}
            assert gs.isdisjoint(groups)
            groups.update(gs)
            with (out/f"{task}_{role}.jsonl").open("w",encoding="utf8") as f:
                for k in ids:
                    v=rows[k]
                    if role=="history":v={key:value for key,value in v.items() if key!="new"}
                    else:assert "new" in v
                    if truth and role=="query":assert truth[task,k]==(v["group"],v["new"])
                    f.write(json.dumps(v,ensure_ascii=False)+"\n")
            summary[task][role]=len(ids)
        if truth:assert set(roles[task]["query"])=={k for d,k in truth if d==task}
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
