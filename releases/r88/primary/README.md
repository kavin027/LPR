# Numerical replay of legacy-profile reranking

Extract the complete ZIP and run `python -B verify.py` from that directory. This command checks all payload hashes, joins predictions by request ID, rebuilds the numerical comparisons and runs each module below. Python 3, NumPy and pandas suffice. The clean check used Python 3.13, NumPy 2.2.1 and pandas 2.3.1. One CPU thread per numerical library gives predictable runtime. The optional threadpoolctl package limits thread counts when present.

## Entry points

Run any of these commands separately for a narrower review.

| Command | What it checks |
| --- | --- |
| `python -B mapping/verify.py` | Fold-wise selection of the direct mapping on the common base, and the incremental count and profile terms |
| `python -B reselection/verify.py` | Independent weight selection within each of 40 alternative registries and paired performance on repeated requests |
| `python -B mapping/refit_mapping.py` | Refit the mapping from dual annotations and reproduce its held-fold and query decisions |
| `python -B evidence/verify_evidence.py` | Whole-group paired intervals and the supervised-versus-permuted lexical contrast |
| `python -B registry_votes/verify.py` | Which predictions are registered sets, how often retrieved outputs match the five displayed examples, and how often mapping and LPR are independently correct |
| `python -B interface/verify_interface.py` | Heterogeneous old classifier and direct predicted/gold-input mapping, including exact-set counts and selected weights |
| `python -B decoder/verify_refits.py` | Registry-fitted comparator weights, including ABC and PCC, the text-only classifier and scorer selection |
| `python -B stats/verify_statistics.py` | Paired intervals, partial-credit outcomes, sensitivity and transfer of an original registry's weights |
| `python -B common_proposals/verify_common.py` | All 90 unit-weight task, base and method accuracies and 14,133 inverse-only / Base + Inverse prediction pairs |
| `python -B candidate_budget/candidate_budget.py` | Four SpiderOps proposal budgets, including agreement with every original five-flip prediction |
| `python -B icl/verify_icl.py` | The two constrained text-model references, legal candidate membership and paired comparisons |

## Inputs and scope

The compact package starts at frozen candidate lists, score arrays, held-out decisions or predictions, as appropriate for each analysis. The verifier recomputes exact-set accuracy from targets and predictions, the selected grid values from the held-out registry decisions, and per-group paired intervals from query IDs. The separate feature-level archive refits the statistical components from their packaged features. Its default entry point reuses neural logits and language-model generations. The two archives expose different starting points for replication.

`primary/` contains ordered candidate masks, label scores, the profile Gram matrix and the test projection. `cal/`, `metric/`, `placebo/` and `additive/` contain grouped calibration ledgers and component controls. `cohort/` and `redraw/` retain the original draw comparisons. The compact package retains all three query cohorts and the archived ordered IDs. Include missed proposals in the exact-set denominator. Resample entire dialogues or databases for paired intervals. Registry draws overlap on the evaluation requests.

`manifest.json` contains one SHA-256 checksum per payload file. `python -B verify.py --seen-output seen.csv` also exports seen-cohort transitions from the calibrated and fitted-additive controls.
