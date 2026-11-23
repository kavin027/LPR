"""Rebuild the three central analyses and compare every output against its reference."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ANALYSES = (
    ("review7_statistics", "statistics.py"),
    ("review7_selector", "replay.py"),
    ("review8_statistics", "statistics.py"),
)


def main():
    with tempfile.TemporaryDirectory(prefix="crossed-replay-") as temp:
        for directory, script in ANALYSES:
            source = ROOT / directory
            output = Path(temp) / directory
            subprocess.run(
                [sys.executable, str(source / script), str(ROOT), str(output)],
                cwd=ROOT,
                check=True,
            )
            expected = sorted(p.name for p in source.glob("*.csv"))
            actual = sorted(p.name for p in output.glob("*.csv"))
            if not expected or expected != actual:
                raise ValueError(f"{directory}: expected {expected}, produced {actual}")
            for name in expected:
                pd.testing.assert_frame_equal(
                    pd.read_csv(source / name),
                    pd.read_csv(output / name),
                    atol=1e-8,
                    rtol=1e-8,
                )
            print(f"{directory}: verified {len(expected)} tables", flush=True)


if __name__ == "__main__":
    main()
