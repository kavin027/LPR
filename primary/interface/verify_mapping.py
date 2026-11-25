"""Compare primary DVLC LPR with soft-input direct mapping on identical requests."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def compare():
    data = pd.read_csv(HERE / "predictions.csv.gz", dtype={"id": str, "group": str})
    assert len(data) == 4711 and not data.duplicated(["task","id"]).any()
    results = []
    for task, rows in data.groupby("task"):
        for cohort, subset in [("all", rows), ("unseen", rows[rows.stratum == "unseen_combination"])]:
            groups = sorted(subset.group.unique(), key=int if task == "cn" else str)
            codes = pd.Categorical(subset.group, categories=groups).codes
            a = subset.full_selected.eq(subset.gold)
            b = subset.mapping.eq(subset.gold)
            delta = a.astype(int) - b.astype(int)
            sizes = np.bincount(codes)
            margins = np.bincount(codes, weights=delta).astype(int)
            k = len(groups)
            w = np.random.default_rng(251121).multinomial(k, np.full(k, 1/k), 2000)
            lo, hi = np.percentile(100 * (w @ margins) / (w @ sizes), [2.5, 97.5])
            results.append((task, cohort, len(subset), k, int(a.sum()), int(b.sum()),
                            int((a & ~b).sum()), int((b & ~a).sum()), 100 * delta.mean(), lo, hi))
    return pd.DataFrame(results, columns="task cohort n groups lpr_hits mapping_hits fixes harms gain low high".split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    result = compare()
    path = HERE / "paired_mapping.csv"
    if parser.parse_args().write:
        result.to_csv(path,index=False)
    else:
        pd.testing.assert_frame_equal(result, pd.read_csv(path), atol=1e-9, rtol=1e-9)
    print(result.round(3).to_string(index=False))

