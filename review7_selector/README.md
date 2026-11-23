# Can the registry switch off the inverse adjustment?

The original five-fold grouped selector compared 96 settings that retain inverse scoring (four metrics × four positive inverse weights × six count weights). Another six settings varied only the count weight. This sensitivity compares all 102 settings by their **registration-only** exact-match hits. It keeps the original winner order within each family and places the original hybrid settings first in cross-family ties. The protocol also reports the opposite cross-family tie policy. All comparisons are retrospective.

## What changes when count-only becomes eligible?

Among 42 registries, count-only has a strictly higher maximum registration score on two, ties on one and scores lower on 39. Both strictly higher scores occur on Spider. Including the original registry, each corpus has 21 registries; Spider's additional databases reuse its registry choice. These are 42 selections and 63 cohort evaluations.

| Spider registry | Registration hit advantage of best count-only over best hybrid | Original Spider hits with hybrid → joint, all / unseen | Additional Spider hits with hybrid → joint, all / unseen |
|---|---:|---|---|
| Uniform draw 4 | +2 | 506 → 420 / 40 → 3 | 55 → 41 / 8 → 0 |
| Uniform draw 9 | +1 | 379 → 328 / 17 → 4 | 44 → 40 / 1 → 0 |
| Enriched draw 10 | 0 | 382 → 382 / 83 → 83 | 48 → 48 / 9 → 9 |

If count-only wins the cross-family tie on enriched draw 10, original Spider all/unseen hits become 397/9 and additional Spider 50/3. Both original registries retain exactly the original hybrid decision under either policy. These outcomes provide no evidence that the two switches identified a better unseen rule.

## How much can the within-family tie rule change?

The unit of a tie is a *configuration*. Two winners may share the same metric and use different weights. On the original CrossWOZ registry, six hybrid configurations tie at 43/75; the next distinct hit count is 42. The corresponding Spider values are six at 18/75 and a next distinct 17. For each original registry the gap from the winning configuration to the runner-up configuration is therefore zero. Across all registries, hybrid had multiple winners on 11/21 CrossWOZ and 14/21 Spider registries, with median tie counts 2 and 3. The gap to the next distinct score was one hit on 20/21 and 19/21; the remaining gaps were two hits.

| Original query cohort | Reference (fixed full), all / unseen | Published hybrid, all / unseen | Range of correct counts among registry-tied hybrid winners, all / unseen |
|---|---|---|---|
| CrossWOZ | 1,698 / 612 | 1,806 / 603 | 1,751–1,810 / 576–626 |
| Spider | 444 / 50 | 430 / 52 | 430–470 / 48–59 |
| Additional Spider | 60 / 8 | 57 / 8 | 46–60 / 3–9 |

The observed sign against fixed full can change across the original six tied winners on every unseen cohort. These ranges use query outcomes for description after scoring. They are unavailable at registration and do not estimate the accuracy of an oracle tie rule. They compare systematically interchangeable registration scores, rather than adding independent experimental cohorts.

## Numerical reproduction

From the extracted anonymous evidence archive, run:

    python review7_selector/replay.py . replay-selector-sensitivity
    python review7_selector/verify.py . replay-selector-sensitivity

The replay imports the released scoring and registry code in review5_selector and round2_draws, checks the original winner and number of ties from the archived registration hit grid, and computes scores for all co-winners. Gold labels enter after decoding. All previously published selected, count-only and hybrid masks are checked for exact equality (102,900 query/registry records). No new neural inference, folds or fitted hyperparameters were introduced. The linear maps and covariance are reconstructed using the archived scorer for each support set.

- registry.csv gives every search family's selected parameters, top hit count, gap to runner-up, gap to the next distinct score and all winning metrics.
- tied_options.csv gives all co-winning parameter settings and their paired fixes and harms against fixed full, for each cohort and stratum. The retained selected hybrid remains tie_order = 1.
- tied_ranges.csv gives query-observed minimum, maximum, distinct exact-hit totals and distinct prediction vectors among the registration-tied choices.
- joint_effects.csv supplies fixed-full, fixed-count, selected-count and hybrid comparisons under each cross-family tie policy, including the rows where selections do not change.
