"""Export aligned, frozen r87 predictions for a retrospective index diagnostic."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def read(base, rel):
    table = pd.read_csv(base / rel, dtype={"id": str})
    out = table.set_index(["task", "id"])
    assert out.index.is_unique
    return out

def main(base):
    interface = read(base, "interface/predictions.csv.gz")
    cal = read(base, "cal/predictions.csv.gz")
    metric = read(base, "metric/results/predictions.csv.gz")
    mapped = read(base, "mapping/predictions.csv.gz")
    unsupported = read(base, "cohort/unsupported_targets.csv")
    for t in (cal, metric, mapped):
        assert set(t.index) == set(interface.index)
    out = {}
    for task in ("cn", "spider"):
        labels = np.load(base / f"primary/{task}.npz")["labels"].astype(str).tolist()
        y = np.load(base / f"mapping/inputs/{task}.npz")["y"]
        assert y.shape == (75, len(labels)) and np.isin(y, [0, 1]).all()
        bits = 2 ** np.arange(len(labels))
        rows = []
        for key, row in interface.loc[[task]].iterrows():
            c, m, u = cal.loc[key], metric.loc[key], mapped.loc[key]
            assert row.full_selected == c.full_selected == m.full_selected == u.primary
            assert row.base_forward == c.bf_fixed == c.bf_selected == u.unary
            assert row.mapping == u.map_only
            assert row.gold == m.gold == u.gold and row.stratum == m.stratum == u.stratum
            gold = int(row.gold)
            if row.stratum == "unseen_atom":
                known = unsupported.loc[key, "labels"].split("|")
                assert gold < 0 and set(known) - set(labels)
                gold = sum(1 << j for j, lab in enumerate(labels) if lab in known)
            assert gold >= 0
            rows.append([key[1], row.stratum, gold, int(row.base_forward), int(c.ecp_selected),
                         int(c.row_selected), int(m.isotropic_selected), int(row.mapping),
                         int(u["map"]), int(row.full_selected)])
        out[task] = dict(labels=labels, registry_masks=(y @ bits).astype(int).tolist(), records=rows)
    with gzip.open(HERE / "index_inputs.json.gz", "wt", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("r87_compact", type=Path)
    main(p.parse_args().r87_compact)
