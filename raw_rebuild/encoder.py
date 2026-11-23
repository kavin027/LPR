"""Encode rebuilt task inputs; each output row corresponds to the input row order."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rebuild import HERE,load,sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path,help='downloaded Qwen3-Embedding-0.6B directory')
    parser.add_argument('output',type=Path)
    parser.add_argument('jsonl',type=Path,nargs='+',help='one or more rebuilt files in desired order')
    args=parser.parse_args()
    for k,v in load(HERE/'sources.json')['weights'].items():
        if sha(args.model/k)!=v:raise ValueError('weight digest mismatch: '+k)
    texts=[];row_ids=[]
    for path in args.jsonl:
        for line in path.read_text(encoding='utf8').splitlines():
            row=json.loads(line);texts.append(row['text']);row_ids.append(row['id'])
    model=AutoModel.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(args.model,padding_side='left',local_files_only=True)
    order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
    vectors=np.empty((len(texts),1024),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(order),64):
            ids=order[start:start+64]
            batch=tok([texts[i] for i in ids],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            last=model(**batch,use_cache=False).last_hidden_state[:,-1].float()
            vectors[ids]=torch.nn.functional.normalize(last,dim=1).cpu().numpy()
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    np.savez_compressed(args.output,ids=row_ids,vectors=vectors)

if __name__=='__main__':main()
