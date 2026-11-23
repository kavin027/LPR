"""Freeze registration-only nested paths and query IDs without reading evaluation targets."""
import hashlib, io, json, os
from pathlib import Path
from zipfile import ZipFile
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('BUDGET_EVIDENCE_OUTPUT', HERE))
BASE=Path(os.environ.get('BUDGET_EVIDENCE_INPUT', HERE.parents[2]/'manuscripts/dasfaa2027/review-reproduction.zip'))

def read_np(z, name):
    return np.load(io.BytesIO(z.read(name)))

def main():
    doc=dict(protocol="Original 75 retained in sequence, then request-ID SHA256 order; five fixed paths (1..5). Both endpoints identical across paths. Same feature encoder and legacy service. All-query exact-set accuracy primary; baseline-75 unseen IDs fixed, dynamic unseen separate. Main methods B-only threshold .5, old-score calibrated forward threshold .5, full inverse alpha=1. Auxiliary B-only threshold chosen on registry by leave-one-group-out exact match from .1 to .9 (ties closest to .5, then higher). Compare all three pairwise on identical requests. Endpoint intervals resample dialogues/dbs, 2000, seed 251121. No selection on the test. Eight Spider databases separately evaluated using development fits.", datasets={})
    with ZipFile(BASE) as z:
        roles=json.loads(z.read('raw_rebuild/roles.json'))
        for ds in ['cn','spider']:
            data=json.loads(z.read(f'registration/{ds}_rows.json'))
            reg=data['register']; by_id={v['id']:v for v in reg}
            man=json.loads(z.read(f'registration/{ds}_manifest.json'))
            assert [v['id'] for v in reg]==man['repeat_pool'] and len(reg)==len(by_id)
            groups={v['group'] for v in reg}
            with read_np(z,f'refit_inputs/{ds}.npz') as orig:
                anchor=orig['register_ids'].tolist()
                assert len(set(anchor))==75 and set(anchor)<=set(by_id)
            cohorts={}
            for tag in (['cn'] if ds=='cn' else ['spider','confirmation']):
                with read_np(z,f'refit_inputs/{tag}.npz') as inp:
                    ids=inp['ids'].tolist(); qg=set(inp['query_groups'].tolist())
                    assert not set(ids)&set(by_id) and not qg&groups and len(set(ids))==len(ids)
                    cohorts[tag]=dict(ids=ids,groups=len(qg))
            sizes=sorted({n for n in [75,150,300,600,1200,1622,2400,len(reg)] if n<=len(reg)})
            rest=sorted(set(by_id)-set(anchor))
            paths={str(i):anchor+sorted(rest,key=lambda r:hashlib.sha256(f'B-label-curve-v1|{ds}|{i}|{r}'.encode()).digest()) for i in range(1,6)}
            doc['datasets'][ds]=dict(n=len(reg), groups=len(groups), budgets=sizes, paths=paths, cohorts=cohorts)
    dest=OUT/'freeze.json'
    encoded=(json.dumps(doc,indent=1)+'\n').encode()
    if dest.exists(): assert dest.read_bytes()==encoded
    else: dest.write_bytes(encoded)
    print(hashlib.sha256(encoded).hexdigest())
if __name__=='__main__': main()
