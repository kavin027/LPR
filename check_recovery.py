"""Check the sufficient recovery condition against fixed, unit-weight scores."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root):
    results = {}
    for task in ("cn", "spider"):
        with (root / "controls" / f"{task}_requests.csv").open("rb") as stream:
            requests = pd.read_csv(stream, dtype={"id": str}).set_index("id", verify_integrity=True)
        with (root / "scores" / f"{task}_scoring.npz").open("rb") as stream:
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as data:
                ids = data["ids"]
                weights = data["whitened_weights"]
                observations = data["whitened_observation"]
                labels = data["labels"]
        with (root / "scores" / f"{task}_candidates.npz").open("rb") as stream:
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as data:
                assert np.array_equal(ids, data["ids"]) and np.array_equal(labels, data["labels"])
                offsets, masks, forward = (data[k] for k in ("offsets", "mask", "forward"))
                combined = forward + data["unary"] + data["interaction"]
        assert set(ids) == set(requests.index)
        bits = 1 << np.arange(len(labels))
        counts = {part: dict(reachable=0, full_radius_certified=0, certified=0, certified_forward_misses=0)
                  for part in ("all", "originally_unseen")}
        for i, key in enumerate(ids):
            row = requests.loc[key]
            span = slice(offsets[i], offsets[i + 1])
            options = masks[span]
            matches = np.flatnonzero(options == row.gold)
            if len(matches) == 0:
                continue
            assert len(matches) == 1
            target = matches[0]
            rivals = options != row.gold
            profiles = ((options[:, None] & bits) != 0).astype(float) @ weights
            residual = observations[i] - profiles[target]
            delta = profiles[target] - profiles
            distances = np.linalg.norm(delta, axis=1)
            shared = forward[span][target] - forward[span] + distances**2 / 2
            lower = shared - np.abs(delta @ residual)
            full_lower = shared - distances * np.linalg.norm(residual)
            actual = combined[span][target] - combined[span]
            assert np.all(lower >= full_lower - 1e-8)
            assert np.all(actual[rivals] >= lower[rivals] - 1e-8)
            certified = bool(np.all(lower[rivals] > 1e-8))
            if certified:
                assert int(row.inverse) == int(row.gold)
            for part in (["all", "originally_unseen"] if row.stratum == "unseen_combination" else ["all"]):
                counts[part]["reachable"] += 1
                counts[part]["full_radius_certified"] += int(np.all(full_lower[rivals] > 1e-8))
                counts[part]["certified"] += int(certified)
                counts[part]["certified_forward_misses"] += int(certified and np.any(forward[span] > forward[span][target] + 1e-8))
        results[task] = counts
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Reproduction ZIP or its extracted root")
    args = parser.parse_args()
    if args.archive.is_dir():
        result = audit(args.archive)
    else:
        with zipfile.ZipFile(args.archive) as archive:
            result = audit(zipfile.Path(archive))
    print(json.dumps(result, indent=2))
