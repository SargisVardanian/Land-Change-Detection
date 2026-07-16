# QCPR v3 mask-only diagnostic contract

## Scientific status before this revision

Job `101315` is classified as:

- runtime: `ENGINEERING_PASS`;
- anti-empty-collapse: `PASS`;
- localization: `INCONCLUSIVE`;
- query specificity: `NOT_TESTED`;
- overall scientific status: `SCIENTIFIC_HOLD`.

The last-five foreground probability increased from 0.287 to 0.389, but
background increased from 0.280 to 0.392. The foreground-background margin
therefore changed from +0.0068 to -0.0028, Dice was nearly flat, and IoU
decreased. Stochastic batches make the isolated final step non-comparable.

## Exact data contract

`mask_only_diagnostic` accepts only rows satisfying all three predicates:

```text
dataset_name == s2looking
seg_supervision_mode == query_specific
retrieval_supervision == false
```

The run persists the source manifest SHA256, every filtered pair ID, the fixed
train IDs, and the disjoint fixed validation IDs in
`fixed_probe_contract.json`.

## Optimization contract

Trainable modules are restricted to the generic temporal patch field,
token-patch grounding decoder, and multi-scale mask decoder. Global retrieval,
text adapter, local projection, rerank weights, and region slots remain frozen.
Only supervised mask loss is active.

The three pre-registered objectives are:

| ID | Objective |
|---|---|
| A | Dice + foreground/background-normalized focal (0.75/0.25) |
| B | A + Tversky (alpha=0.3, beta=0.7) |
| C | B with stronger negative focal weight 0.50 |

All objectives use the same seed, eight fixed training rows, sixteen disjoint
validation rows, initialization checkpoint, and evaluation schedule.

## Verified mismatch and query-swap contract

S2Looking appeared/disappeared rows sharing one base pair are linked. A wrong
direction query is recomputed through a separate model forward. It contributes
an empty-target loss only when the opposite directional ground-truth mask is
actually empty. The mismatch term is included exactly once.

For query specificity, both directional queries are evaluated on identical
T1/T2 images. The report records correct-query IoU, swapped-query IoU, and
their gap without treating a non-empty opposite mask as an empty negative.

## Scientific metrics and gate

Validation occurs at step 0 and every five steps on the same samples. Dice,
IoU, precision, and recall are unsmoothed. The report also records micro
metrics, PR-AUC, foreground/background probabilities and margin, target and
predicted area, samples with true positives, selected near-empty rate,
negative-pair active rate, and empty-target false-positive area.

A 100-step C1 diagnostic is forbidden unless one 20-step micro-overfit
objective improves foreground probability, foreground-background margin by at
least 0.05, Dice, and IoU; retains precision; has true positives on at least
75% of non-empty validation samples; avoids near-empty collapse; does not
worsen empty-mask false positives; and obtains a positive query-swap gap.
Training loss is not a selection criterion.
