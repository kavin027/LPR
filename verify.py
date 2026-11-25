"""Verify all published checks and regenerate the inputs to the main results tables."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

root = Path(__file__).resolve().parent
if not __debug__:
    raise SystemExit('Use ordinary Python, with assertions enabled.')
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')

def run(name, *args):
    script=root/name
    subprocess.run([sys.executable,'-B',str(script),*map(str,args)],cwd=script.parent,env=env,check=True)

def compare(actual, published):
    left=pd.read_csv(actual)
    right=pd.read_csv(published)
    pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-10,atol=1e-10)

for name, expected in json.loads((root/'hashes.json').read_text(encoding='utf8')).items():
    with (root/name).open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==expected,name

run('primary/verify.py')
for subdir in ('ext-cal','matched','neural'):
    run('extensions/'+subdir+'/verify.py','--reference',root/'primary')
for check in ('joint_mapping/verify.py','registration/verify.py','registration/perturb.py',
              'index_workload/verify_index_workload.py','extensions/density/verify.py','extensions/open_language_model/verify.py'):
    run(check)
run('analysis/retrieval/retrieval.py',root)
with tempfile.TemporaryDirectory() as temporary:
    out=Path(temporary)
    for source,args,files in [
        ('registry/registry.py',[root/'registration/standalone',out,'--no-plot'],['registry-draws.csv','registry-means.csv']),
        ('base_comparison/build_multibase.py',[root,out],['base-scores.csv']),
        ('base_comparison/compare_mapping.py',[root,out/'unit-vs-mapping.csv'],['unit-vs-mapping.csv'])]:
        run('analysis/'+source,*args)
        for file in files:
            compare(out/file,root/'analysis'/Path(source).parent/file)
print('PASS: numerical replication complete.')
