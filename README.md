# Legacy Profile Reranking

This numerical replication package accompanies *Legacy Profile Reranking for Multi-Label Classification after a Label Schema Change*. It contains candidate scores, registration folds, evaluation masks, paired predictions, and the code used to recover the reported results on **2,744 CrossWOZ** and **1,967 SpiderOps** requests.

## Run

From the root of an extracted download or a clone of this branch, use Python 3.11+ with NumPy and pandas. Run the verifier under ordinary Python with assertions enabled.

```sh
python -m pip install -r requirements.txt
python -B verify.py
```

The command verifies the file manifest, recomputes the numerical comparisons, checks the derived tables against frozen results, and prints `PASS: numerical replication complete.` It writes temporary comparison files outside the package. No network or model server is used. For the optional vector figure, install Matplotlib and run:

```sh
python analysis/registry/registry.py registration/standalone figure-output
```

## Evidence guide

| Question | Code and input data |
|---|---|
| How do candidate scores from different external classifiers compare? | `analysis/base_comparison/` and `primary/bases/` reconstruct base scores and LPR on identical proposals. |
| What do external chain, correlation and clustering adaptations contribute? | `extensions/ext-cal/` verifies grouped selections and predictions for T5 Chain and SCOTTA. `primary/decoder/` includes ABC and PCC. |
| What does a direct mapping contribute? | `primary/mapping/` refits the mapping from paired annotations; `joint_mapping/` compares each of the 25 joint coefficient choices on held-out groups. |
| How do 20 registrations change the result? | `registration/` replays the stored holdout and query grids. `analysis/registry/` calculates the two cohorts and draw-specific accuracy differences, including the fixed unit coefficient. |
| How do index lookups change? | `index_workload/` reconstructs SQLite postings and returned record IDs. `analysis/retrieval/` verifies macro and micro F1, micro precision and recall, paired intervals, and posting changes for registered single-label and two-label lookups. |
| How do the metric and historical observations matter? | `extensions/density/` refits held-out profile residuals and checks 450 Gaussian log-density values. `primary/evidence/` and `primary/interface/` check the lexical supervision placebo. `extensions/matched/` recomputes a base on the Forward encoder. |
| What do the text-generation and neural runs establish? | `extensions/open_language_model/` checks the parsed outputs, correctness, format failures and paired intervals. `extensions/neural/` replays saved neural out-of-fold and query predictions. |

In all directories, a predicted nonnegative integer is a bit mask on that task's revised label inventory. Exact-set accuracy requires agreement on every label. A negative sentinel marks an unavailable target or an invalid prediction as specified by the corresponding verifier. The language-model check starts from parsed output masks. Model refitting or regeneration from the original text is outside this numerical release.
