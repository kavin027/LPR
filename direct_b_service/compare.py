"""Replay the archived A construction on B annotations. No legacy scores enter B-only arms."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def fit_service(archive, folder, inputs):
    source = folder / "historical_readout.py"
    source.write_bytes(archive.read("raw_rebuild/historical_readout.py"))
    vectors = np.concatenate([inputs["s"], inputs["t"]])
    ids = np.concatenate([inputs["register_ids"], inputs["ids"]])
    assert len(set(ids)) == len(ids)
    assert not set(inputs["groups"]) & set(inputs["query_groups"])
    np.savez(folder / "vectors.npz", ids=ids, vectors=vectors)
    labels = inputs["labels"]
    examples = [
        {"id": str(key), "old": labels[row.astype(bool)].tolist()}
        for key, row in zip(inputs["register_ids"], inputs["y"], strict=True)
    ]
    records = folder / "B_labels.jsonl"
    records.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in examples), encoding="utf8")
    dest = folder / "scores.npz"
    subprocess.run([sys.executable, str(source), str(folder / "vectors.npz"), str(records), str(dest)], check=True)
    with np.load(dest) as out:
        indices = [out["old_labels"].tolist().index(label) for label in labels]
        assert np.array_equal(out["ids"], ids[len(examples):])
        return out["old_scores"][:, indices]


def registry_threshold(embedding, target, groups):
    # Linear dual of the archived unit-penalty, centered primal ridge.
    held_scores = np.empty_like(target, dtype=float)
    for group in np.unique(groups):
        held = groups == group
        x, y = embedding[~held], target[~held]
        center = x.mean(axis=0)
        xc = x - center
        dual = np.linalg.solve(xc @ xc.T + np.eye(len(x)), y-y.mean(axis=0))
        held_scores[held] = y.mean(axis=0) + (embedding[held]-center) @ xc.T @ dual
    clipped = np.clip(held_scores, 0, 1)
    grid = range(1, 10)
    hits = {t:int(np.all((clipped >= t/10) == target, axis=1).sum()) for t in grid}
    chosen = max(grid, key=lambda t:(hits[t], -abs(t-5), t))
    return chosen/10, hits


def decide(scores, y):
    bits = 1 << np.arange(y.shape[1])
    base = (scores >= .5).astype(int) @ bits
    eps = 1 / (len(y) + 2)
    bounded = np.clip(scores, eps, 1 - eps)
    known = np.unique(y.astype(int) @ bits)
    counts = np.bincount(y.sum(axis=1).astype(int), minlength=y.shape[1]+1)
    k = min(5, len(bits))
    flips = (np.arange(1 << k)[:, None] & (1 << np.arange(k))) != 0
    count_adjusted = []
    for p, initial in zip(bounded, base, strict=True):
        near = np.argsort(abs(p-.5), kind="stable")[:k]
        options = np.union1d(known, initial ^ (flips @ bits[near]))
        options = np.r_[initial, options[options != initial]]
        masks = (options[:, None] & bits) != 0
        forward = masks @ np.log(p/(1-p))
        assert options[np.argmax(forward)] == initial
        adjusted = forward + np.log1p(counts[masks.sum(axis=1)])
        count_adjusted.append(options[np.argmax(adjusted)])
    return base, count_adjusted


def summarize(frame, dataset):
    rows = []
    for population, q in [("all", frame), ("unseen",frame[frame.stratum=="unseen_combination"]), ("seen",frame[frame.stratum=="seen"])]:
        codes, ids = pd.factorize(q.group, sort=True)
        gen = np.random.default_rng(251121)
        boot = gen.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=2000)
        den = boot @ np.bincount(codes)
        for method in ["b_only","b_only_cv","b_only_count","forward","fixed_size","inverse"]:
            correct = q[method].eq(q.gold)
            for reference in ["b_only","b_only_cv","b_only_count","forward"]:
                anchor = q[reference].eq(q.gold)
                net = correct.astype(int)-anchor.astype(int)
                delta = 100 * net.mean()
                lo, hi = np.percentile(100*(boot @ np.bincount(codes, weights=net))/den,[2.5,97.5])
                rows.append(dict(dataset=dataset, population=population, method=method, reference=reference, n=len(q), hits=int(correct.sum()), accuracy=100*correct.mean(), fixes=int((correct & ~anchor).sum()), harms=int((anchor & ~correct).sum()), delta=delta, lower=lo, upper=hi))
    return rows


def main(package, out):
    rows = []
    all_queries = []
    calibration = []
    with zipfile.ZipFile(package) as archive, tempfile.TemporaryDirectory() as temp:
        for tag in ["cn","spider","confirmation"]:
            folder = Path(temp) / tag
            folder.mkdir()
            with np.load(io.BytesIO(archive.read(f"refit_inputs/{tag}.npz"))) as x:
                assert len(x["y"]) == 75
                if tag == "spider":
                    spider_registry = {k:x[k] for k in ["s","y","labels","register_ids"]}
                if tag == "confirmation":
                    for k, val in spider_registry.items():
                        np.testing.assert_array_equal(x[k], val)
                scores = fit_service(archive, folder, x)
                pred, count_pred = decide(scores, x["y"])
                threshold, hits = registry_threshold(x["s"], x["y"], x["groups"])
                calibration.append(dict(dataset=tag, threshold=threshold, fold_hits=hits))
                pred_cv = (scores >= threshold).astype(int) @ (1 << np.arange(scores.shape[1]))
                frame = pd.read_csv(archive.open(f"controls/{tag}_requests.csv"), dtype={"id":str,"group":str}).set_index("id")
                assert frame.index.is_unique and set(frame.index) == set(x["ids"])
                frame = frame.loc[x["ids"]].copy()
                np.testing.assert_array_equal(frame.group, x["query_groups"].astype(str))
                frame["b_only"], frame["b_only_count"] = pred, count_pred
                frame["b_only_cv"] = pred_cv
                frame["dataset"] = tag
                rows.extend(summarize(frame, tag))
                all_queries.append(frame[["dataset","group","stratum","gold","b_only","b_only_cv","b_only_count","forward","fixed_size","inverse"]])
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf8")
    pd.concat(all_queries).to_csv(out / "requests.csv")
    pd.DataFrame(rows).to_csv(out / "results.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(1):
        main(args.package,args.output)
