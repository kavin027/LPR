# Legacy Profile Reranking — numerical review release r88

Download this release directory with its subdirectories intact. Using Python 3.11 or later, NumPy and pandas, run `python -B verify.py` from this folder. The verifier checks payload hashes and recomputes the registered choices and paired comparisons. Neural retraining and a model server are unnecessary. The main cohorts contain 2,744 CrossWOZ and 1,967 SpiderOps requests. Main-paper table numbers refer to this revision.

| Paper result | Files and unit of verification |
|---|---|
| Table 3, historical interfaces and mapping | `primary/interface`, `primary/mapping`, `extensions/ext-cal`, `mapping88`. Ordered candidates, original predictions, five-fold exact hits and all 25 joint choices. |
| Table 4 and index retrieval | `primary/stats`, `index88`. Gold/predicted masks, registry predicates, retrieved IDs, empty-retrieval policy and SQLite intersection checks. |
| Table 5, metric and weight | `primary/decoder`, `primary/cal`, `extensions/density`. Registry group splits, held-out hits, metric choices and predictions. The density calculation is in `extensions/feature-rebuild/covariance.py`. |
| Weight transfer and Table 6, registration draws | `primary/redraw`, `primary/reselection`, `primary/stats`, `registry88`, `extensions/neural`. Registry IDs, source weights, receiving fits, draw-specific cohorts, paired gains and a conditional perturbation analysis. |
| Table 7, observation controls | `primary/placebo`, `primary/evidence`, `extensions/matched`. Observation protocols, paired predictions and independent additive controls in `primary/additive`. |
| Text generation | `extensions/open_language_model/results` contains derived prediction masks and parsing statistics. Raw prompts and generated responses are distributed only in the separate offline author package. |

The primary directory preserves the frozen numerical package for the preceding revision. Its hash manifest is kept intact. The additional mapping, registry and posting-list analyses are retrospective. The per-directory READMEs document additional checks and which inputs are frozen. The numerical packet contains neither raw prompts nor literal generated answers.
