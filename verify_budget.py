"""Replay the B-label budget experiment using only the extracted archive."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
CURVE = "b_annotation_curve"
ANCHOR = "direct_b_service"


def compare_outputs(reference, actual, files):
    for name in files:
        left, right = reference / name, actual / name
        if name.endswith(".json"):
            assert json.loads(left.read_bytes()) == json.loads(right.read_bytes()), name
        else:
            types = {"id": str, "group": str}
            expected = pd.read_csv(left, dtype=types)
            replay = pd.read_csv(right, dtype=types)
            pd.testing.assert_frame_equal(replay, expected, check_exact=False, atol=1e-10, rtol=1e-10, obj=name)
        print(f"Verified {reference.name}/{name}", flush=True)


def main():
    names = ["raw_rebuild/roles.json", "raw_rebuild/historical_readout.py",
             "round2_draws/refit_draws.py", "registration/confirmation_gold.json"]
    for tag in ("cn", "spider"):
        names.extend(f"registration/{tag}_{suffix}" for suffix in ("manifest.json", "rows.json", "panel.npz"))
    for tag in ("cn", "spider", "confirmation"):
        names.extend([f"refit_inputs/{tag}.npz", f"controls/{tag}_requests.csv"])
    with TemporaryDirectory(prefix="label-budget-") as directory:
        work = Path(directory)
        input_zip = work / "frozen-inputs.zip"
        with ZipFile(input_zip, "w") as archive:
            for name in names:
                archive.write(ROOT / name, name)
        (work / CURVE).mkdir()
        environment = dict(os.environ, BUDGET_EVIDENCE_INPUT=str(input_zip),
                           BUDGET_EVIDENCE_OUTPUT=str(work / CURVE), PYTHONDONTWRITEBYTECODE="1")
        def run(*args):
            subprocess.run([sys.executable, *map(str, args)], env=environment, check=True, cwd=work)
        run(ROOT / ANCHOR / "compare.py", input_zip, "--output", work / ANCHOR)
        compare_outputs(ROOT / ANCHOR, work / ANCHOR, ("requests.csv", "results.csv", "calibration.json"))
        for script in ("freeze.py", "run.py", "report.py"):
            run(ROOT / CURVE / script)
        compare_outputs(ROOT / CURVE, work / CURVE, ("freeze.json", "fits.json", "predictions.csv.gz",
                        "changes.csv", "curve.csv", "endpoint_intervals.csv", "hits.csv", "gap_change_vs_75.csv"))
    print("Clean replay of the B-budget experiment passed.")


if __name__ == "__main__":
    main()
