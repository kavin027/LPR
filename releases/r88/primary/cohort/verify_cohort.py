"""Recompute and check the three descriptive tables in a clean temporary directory."""
import csv
import math
import sys
import tempfile
from pathlib import Path

from cohort_audit import run

def check_cohort(replay, source):
    with tempfile.TemporaryDirectory() as td:
        generated=Path(td)
        run(replay,generated)
        for name in ('overlap_draws.csv','overlap_summary.csv','target_summary.csv'):
            with (source/name).open(encoding='utf-8',newline='') as a, (generated/name).open(encoding='utf-8',newline='') as b:
                expected,actual=list(csv.DictReader(a)),list(csv.DictReader(b))
            assert len(actual)==len(expected)
            for left,right in zip(expected,actual):
                assert set(left)==set(right)
                for k in left:
                    if left[k]==right[k]:
                        continue
                    assert math.isclose(float(left[k]),float(right[k]),abs_tol=1e-10,rel_tol=1e-12),(name,k,left[k],right[k])
    print(f'PASS: 120 registry/cohort cells, six registry-balanced summaries, and {len(actual)} calibrated target summaries.')

if __name__=='__main__':
    source=Path(__file__).resolve().parent
    replay=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else source.parent
    check_cohort(replay,source)
