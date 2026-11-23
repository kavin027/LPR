"""Run a matched MEKA baseline refit from the exact released ARFF inputs."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile
from shutil import copyfile

ROOT = Path(__file__).resolve().parent


def run(tag, meka):
    cp = str(meka.resolve()/'lib'/'*')
    assert (meka/'lib'/'meka-1.9.9.jar').exists(), 'Expected the extracted MEKA 1.9.9 binary distribution'
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for name in ('support.arff','query.arff','options.txt'):
            copyfile(ROOT/'chain_inputs'/tag/name, d/name)
        subprocess.run(['javac','-encoding','UTF-8','-cp',cp,'-d',str(d),
                        str(ROOT/'controls'/'MatchedEnsemble.java')],check=True)
        subprocess.run(['java','-Xmx1800m','-cp',str(d)+os.pathsep+cp,
                        'MatchedEnsemble',str(d),'16' if tag=='cn' else '11'],check=True)
        with (d/'ensemble.csv').open(newline='') as new, (ROOT/'controls'/f'{tag}_blind.csv').open(newline='') as old:
            current, expected = list(csv.DictReader(new)), list(csv.DictReader(old))
        assert len(current)==len(expected)
        assert all(all(a[k]==b[k] for k in a) for a,b in zip(current,expected))
        assert (d/'orders.txt').read_text().splitlines()==(ROOT/'controls'/f'{tag}_orders.txt').read_text().splitlines()
        print(tag,len(current),'exact decisions: ten matched chains, probability ensemble and independent SMO')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('meka_directory',type=Path,help='Directory containing lib/meka-1.9.9.jar')
    ap.add_argument('tags',nargs='*',default=['cn','spider','confirmation'])
    args=ap.parse_args()
    for tag in args.tags:run(tag,args.meka_directory)
