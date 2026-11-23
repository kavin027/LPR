# Legacy Profile Reranking (LPR)

Experiment code for *Legacy Profile Reranking for Multi-Label Classification after a Label Schema Change*.

LPR predicts sets of labels after a service changes its label schema. Forward proposes sets from text embeddings and the frozen legacy classifier scores. For each candidate set, LPR predicts its expected legacy score vector from a small registry of dual annotations. It subtracts a covariance-weighted mismatch penalty from the forward score and returns the highest-scoring set. In the frozen prediction files, the column called `inverse` (or `full` in some scripts) is the complete LPR prediction.

## Reproduce the main comparison on CPU

From the repository root, use Python 3.13 and install the pinned numerical dependencies. Frozen inputs are included for the original 75-example registries. A GPU and model downloads are optional for these commands.

```sh
python -m pip install -r requirements.txt
python refit.py
python reproduce.py
python check_recovery.py .
```

- **refit.py** fits the Forward and inverse readouts from the registry arrays, generates candidates, and checks the predictions from LPR and the label-count prior for every query. The cohorts contain 2,744 CrossWOZ turns, 1,967 Spider questions aligned with BREAK, and 189 questions from eight additional Spider databases.
- **reproduce.py** checks individual predictions against the recorded sets, then reports fixes, harms, accuracy differences in percentage points and paired 95% intervals on the two development cohorts. PCC here uses frozen predictions. To refit it with MEKA, see below.
- **check_recovery.py** checks the sufficient recovery condition using the evaluation labels after prediction.

Exact-set accuracy on combinations absent from registration whose individual labels were registered:

| Cohort | Requests | Forward | Count prior | Single-chain PCC | LPR | LPR fixes / harms vs. prior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CrossWOZ | 1,223 | 33.85% | 41.62% | 19.22% | 50.04% | 170 / 67 |
| Spider to BREAK | 548 | 1.46% | 2.92% | 7.30% | 9.12% | 47 / 13 |

On the additional databases, 59 questions met this definition. LPR fixed eight and harmed two relative to the count prior. The database-paired 95% interval for the gain includes zero.

## Further experiments

Run the following from the repository root. The budget replay uses the extra dependency listed in requirements-budget.txt.

```sh
python -m pip install -r requirements-budget.txt
python verify_budget.py
python verify_crossed.py
python review6_baselines/baselines.py . outputs/baselines
python review6_baselines/verify.py . outputs/baselines
python round5_interfaces/interfaces.py . outputs/interfaces
```

The budget experiment compares Direct, calibrated Direct, Forward and LPR on identical test requests as the annotation budget increases. The crossed analysis separates changes in the fitted registry from changes in which requests count as unseen. The baseline suite evaluates powerset and pairwise alternatives and replays the CrossWOZ CFPL adaptation from its frozen student scores. These scores come from a single-seed adaptation with a rebuilt teacher and additional source supervision.

With a JDK on PATH and an extracted MEKA 1.9.9 binary distribution, run `python chain_refit.py /path/to/meka-1.9.9` to refit the matched chain baselines from the supplied ARFF inputs.

## Inputs and provenance

The folders refit_inputs, scores and controls hold frozen numeric inputs, candidate scores and request-level outcomes. Fit the readouts using the registration arrays. The gold columns in the outcome files are for evaluation. The paired-evidence.zip file contains the request-level prediction table used by reproduce.py.

To reconstruct source records or regenerate encoder outputs, follow [the raw rebuild instructions](raw_rebuild/README.md). The [source manifest](raw_rebuild/sources.json) records upstream URLs, revisions and checksums for CrossWOZ, the CFPL reannotation, Spider, BREAK and the frozen Qwen encoder. Obtain raw text and weights from the original providers under their respective terms. Frozen-input replay and rebuilding embeddings on different hardware have different numerical guarantees.
