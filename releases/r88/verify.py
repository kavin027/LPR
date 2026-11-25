from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
for name, digest in json.loads((root / "hashes.json").read_text()).items():
    with (root / name).open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == digest, name

checks = [
    ("primary/verify.py", []),
    ("extensions/ext-cal/verify.py", ["--reference", str(root / "primary")]),
    ("extensions/matched/verify.py", ["--reference", str(root / "primary")]),
    ("extensions/neural/verify.py", ["--reference", str(root / "primary")]),
    ("mapping88/verify.py", []),
    ("registry88/verify.py", []),
    ("registry88/perturb.py", []),
    ("index88/verify_index88.py", []),
]
for name, extra in checks:
    path = root / name
    subprocess.run([sys.executable, "-B", str(path), *extra], cwd=path.parent, env=env, check=True)
print("PASS: frozen predictions, registration, mapping and index replay.")
