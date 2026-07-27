# Benchmark Protocol

## Required Separation

The framework should report binary change and semantic change separately.

- binary change answers `did anything change?`
- semantic change answers `what changed into what?`

## Retrieval Metrics

- Recall@1
- Recall@5
- Recall@10
- mAP
- optional transition-consistency score

## Evaluation Splits

Results should not rely only on random splits.

Required holdouts:

- geography holdout
- season holdout

## Hard Negatives

Evaluation should include visually similar but semantically unchanged examples to measure false-change pressure.

## Transition-Centric Evaluation

For semantic-first retrieval, the unit of success is not only visual similarity. It is whether the returned items preserve the transition semantics relevant to the query.
