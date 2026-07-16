# QCPR v3 experiment matrix

Status: active controlled implementation. Smoke, memory, and a 100-step
diagnostic precede any 500-step pilot.

## Fixed protocol

* **Global reference:** a newly trained and accepted clean v3 global-bootstrap
  checkpoint. Historical E0 is unavailable and cannot be substituted silently.
* **Backbone:** the same frozen UniverSat and Jina checkpoints for all rows.
* **Data splits:** frozen natural retrieval validation; frozen localization
  validation; a separately constructed balanced compositional validation slice.
* **Initialization:** frozen pretrained Jina v5 and UniverSat, with randomly
  initialized temporal/global projection and v3 grounding modules under one
  recorded seed. No historical-global-teacher distillation is active.
* **Budget:** same maximum optimization steps, batch, augmentations, and
  scheduler for each ablation; no hidden extra tuning for a preferred row.
* **Seeds:** at least 3 fixed seeds for claims; an initial single-seed pilot is
  labelled diagnostic only.
* **Uncertainty:** paired bootstrap confidence intervals over queries and
  paired ablation differences.  Do not declare improvement from one unpaired
  point estimate.
* **Selection:** checkpoint choice and calibration split are fixed before test
  evaluation. Exact-pair metrics are diagnostic only.

## Metrics and denominators

| Family | Primary measures | Required conditioning |
|---|---|---|
| Candidate generation | semantic R@1/5/10, candidate recall@N | full natural retrieval validation |
| Reranking | end-to-end semantic R@K, conditional R@K given positive in top-N, gain/loss, queries lost before reranking | natural + balanced compositional slices |
| Composition | structured nDCG@K; direction/location/relation accuracy; count MAE and bucket accuracy | human-verified subset primary; parser-derived labels secondary |
| Masks | Dice, IoU, precision, recall, boundary IoU, pointing game | empty/non-empty, query-specific/generic, source dataset separately |
| Temporal | appeared/disappeared accuracy and reversal consistency | only trusted temporal labels |
| Efficiency | encode/rerank time, GPU memory, candidates/sec | same H100 and chunk configuration |

`ECE`/`Brier` may only be named calibration metrics after a logistic/temperature
calibrator is fit on a disjoint calibration split and frozen. Before that,
report them as uncalibrated score diagnostics.

## Ablation ladder

| Row | Change from preceding row | Active training signal | Main hypothesis / preregistered gate | Must not change |
|---|---|---|---|---|
| A | clean v3 global bootstrap | duplicate-aware multi-positive global contrastive + base-text adapter preservation | establish candidate recall@N and natural text-derived-semantic reference; exact pair remains diagnostic | all fixed protocol fields |
| B | A + generic temporal patch late interaction | global + generic local contrastive/retrank loss | conditional reranking gain > 0; no predeclared global degradation | mask decoder, slots, sampler, temporal reversal |
| C | B + query-conditioned multi-scale soft mask | B + Dice + focal/Tversky | faithful mask parity; small-object/boundary quality and/or conditional gain improve vs B | slots, rebalanced sampler, reversal |
| D | C + generic latent region slots | C + optional class-agnostic slot activation/diversity; verified-count loss only where available | count bucket accuracy/MAE and multi-instance masks improve vs C | sampler, reversal |
| E | D + capped rebalanced sampler | D losses, altered train sampling only | balanced-slice gains without material natural-slice regression | architecture, seed set, steps |
| F | E + temporal reversal | E + reversal loss | temporal-direction/reversal measures improve vs E | sampler, architecture, seed set, steps |

No row is a combined “everything changed” pilot.  If a row fails, later rows
are not interpreted as proof that its component works.

## Gates between rows

1. **A → B:** publish candidate recall@{50,100,200}, exact-pair diagnostic and
   clearly labelled text-derived-semantic metrics; if the positive is absent
   too often, improve global retrieval/data first rather than blaming reranking.
2. **B → C:** `conditional reranking gain` must be non-negative within its CI,
   and generic token–patch score must separate positives from audited negatives.
3. **B → C:** mask logits, displayed mask, masked pooled embedding and local
   score must be numerically identical under the canonical scoring API; report
   empty/non-empty masks separately.
4. **C → D:** verified count coverage must meet a predeclared minimum. If not,
   slots can remain for generic region analysis but no count claim is made.
5. **D → E:** sampler audit must show capped weights, duplicate controls and no
   leakage across all split variants.
6. **E → F:** temporal reversal uses only trusted direction labels and must not
   degrade natural retrieval outside the preregistered tolerance.
7. **Any → next:** no NaN/Inf, exact deterministic run configuration, and
   frozen split checks pass.

## Human-verified compositional benchmark

Create a frozen subset of 300–500 queries.  For each query, annotators review
the global top-`N` candidates plus a randomized candidate sample and record:

```text
query_id, pair_id, relevance (yes/no/uncertain),
object evidence, temporal direction, location relation, count bucket,
relation, query-specific mask availability, mask reviewer/source,
annotator_id, confidence, adjudication_status, notes
```

Schema rules:

* two independent annotations for relevance and each applicable attribute;
* adjudication for disagreements; `uncertain` is excluded from the primary
  denominator but retained/audited;
* masks have provenance and never overwrite visual ground truth;
* candidate captions are all visible to annotators, preventing a first-caption
  shortcut; and
* a hash of pair assets, query text, and annotation version freezes the set.

Use this subset to test compositional claims. Parser signatures are useful for
stratification and candidate mining, not a replacement for human relevance.

## Scientific reporting template

Every completed row publishes:

1. exact code/data/checkpoint hashes, branch, seed, hardware and dirty state;
2. fixed-protocol compliance table;
3. natural and balanced-slice results with bootstrap intervals;
4. global candidate recall and conditional reranking decomposition;
5. mask results by empty/non-empty and supervision provenance;
6. qualitative contact sheet sampled before looking at results;
7. failures and a statement of which hypotheses remain unsupported.

## Migration from QCPR v2

1. Historical E0/v2 run directories are missing after cleanup. Preserve all
   remaining artifacts; never treat a new checkpoint as a restored E0.
2. Fork a clean v3 model namespace rather than incrementally modifying the v2
   attribute-head path.
3. First implement rows A/B with a canonical scoring API and exact parity tests.
4. Add C only after generic late interaction is measured; add D only after C.
5. Build/review coverage report and frozen evaluation subsets before E.
6. Add F last. A failure at any gate produces a report, not an automatic larger
   training run.
