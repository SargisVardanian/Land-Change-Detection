# Dataset coverage, audit, and rebalancing plan for QCPR v3

Status: design and audit contract.  This plan does not alter images, masks, or
captions automatically.

## Principles

1. Visual masks and source imagery are immutable ground truth/provenance.
2. Caption parsers may create *audit metadata*, strata, and candidate negatives;
   they never become mandatory v3 model branches.
3. Generated captions/counterfactuals are quarantined with confidence,
   provenance, and human review. They cannot silently join training or test.
4. Rebalancing changes the train sampler only. Natural validation/test keep
   source prevalence; balanced validation is an additional fixed slice.
5. Pair/image/caption duplicates are identified before splitting and weighting.

## Joint coverage table

For every `(pair_id, caption_id)` create a versioned audit row:

```text
dataset, split, pair_id, caption_id, t1_hash, t2_hash, mask_hash,
caption_normalized, caption_source, source_confidence,
changed/no_change, temporal_direction, object_family, location_phrase,
count_bucket, relation, mask_kind, mask_empty, mask_area_fraction,
mask_boundary_length, image_resolution, duplicate_cluster,
near_duplicate_cluster, annotation_status, parser_version
```

The report must tabulate individual attributes and the joint cells below.  The
joint cell is a coverage diagnostic, not an assertion that parser labels are
perfect truth:

\[
cell=(dataset, changed/no\_change, direction, object, location, count,
relation, mask\_kind, area\_bin).
\]

Report counts, unique pairs, unique captions, train/validation/test overlap,
and missingness per cell.  Attribute labels with `unknown` remain explicit;
they are not guessed.

## Duplicate and near-duplicate audit

Perform in this order:

1. exact asset hashes: `(t1_hash,t2_hash,mask_hash)`;
2. normalized-caption duplicates within and across pairs;
3. perceptual image hashes for T1/T2 and a conservative pair-level threshold;
4. optional frozen-backbone embedding nearest neighbours, reviewed as candidate
   clusters rather than automatically merged.

Each cluster report includes member IDs, splits, captions, similarity basis,
distance, and an action `{keep, train-only, review, exclude-from-benchmark}`.
Cross-split duplicate clusters are leakage blockers for the scientific benchmark.

## Sampling weights

Only samples that pass provenance/quality checks are eligible.  With smoothed
cell count `n_c`, quality factor `q_i ∈ [q_min,1]`, exponent `α`, and clipping:

\[
\tilde w_i=q_i(n_{cell(i)}+\tau)^{-\alpha},\qquad
w_i=\operatorname{clip}(\tilde w_i,w_{min},w_{max}),\qquad
\hat w_i=\frac{w_i}{\operatorname{mean}_{train}(w)}.
\]

Pre-register initial conservative candidates `α ∈ {0.3,0.5}`, `τ=5`, and a
weight ratio cap ≤ 5 before normalization.  Choose one setting on a train-only
coverage/calibration procedure; do not optimize it against test metrics.

Additional safeguards:

* cap aggregate mass per duplicate/near-duplicate cluster;
* require a minimum quality/confidence for rare cells;
* do not add a count-specific loss until verified count coverage reaches a
  predeclared threshold;
* log available, selected, unique, and repeat sample counts by cell; and
* inspect effective sampling distribution every epoch.

## Natural and balanced validation subsets

* **Natural validation:** unchanged source distribution, primary estimate of
  expected performance.
* **Balanced compositional validation:** fixed, disjoint pair subset with a
  minimum target per reliable joint stratum; used to diagnose rare concepts.
* **Human-verified benchmark:** separate compositional subset described in
  `qcpr_v3_experiment_matrix.md`; primary for claims about relation/count/location.

No validation subset is resampled at evaluation time. Splits, queries, pair IDs,
and annotation versions receive content hashes.

## Caption/counterfactual governance

An LLM or heuristic can propose duplicate flags, normalized captions,
paraphrases, and natural counterfactual candidates.  Every proposal records:

```text
proposal_id, source_pair/caption, proposed_text, operation,
model_or_rule_version, confidence, rationale, validator_status,
human_review_status, target_split, timestamp
```

It may enter a training augmentation pool only after confidence and human/audit
thresholds are met. It may never replace a visual mask automatically, never
change a held-out label, and never be presented as ground truth.

## Deliverables before rebalanced ablation E

1. `dataset_coverage_report.json` and readable Markdown/CSV tables;
2. duplicate/near-duplicate cluster report and split-leakage decision log;
3. frozen natural/balanced/benchmark manifest hashes;
4. sampler-weight histogram, per-cell mass, clipping statistics and quality
   exclusions;
5. generated-caption/counterfactual quarantine audit; and
6. a manual review protocol with named ownership and adjudication rules.

## Initial audit questions

The first read-only audit answers, per source and split:

* How many unique image pairs versus caption rows exist?
* Which object/direction/location/count/relation combinations have zero,
  singleton, or high coverage?
* Are no-change and empty-mask examples sufficient and visually diverse?
* Do duplicate clusters cross retrieval/localization validation boundaries?
* Is count supervision actually present at a level adequate for an optional
  generic slot experiment?
* Which cells are reliable enough for a balanced slice, and which must remain
  descriptive-only until human verification?

## Read-only baseline audit (2026-07-14)

This first audit read the three current manifests and applied the existing
caption parser **only as an audit tool**. It did not modify a manifest, image,
mask, caption, or sampling weight.

| Dataset | Pair rows | Caption rows | Split rows (train/val/test) | Mask rows | Retrieval-supervised pairs |
|---|---:|---:|---|---:|---:|
| LEVIR-MCI | 10,072 | 50,360 | 6,810 / 1,333 / 1,929 | 10,072 | 10,072 |
| SECOND-CC | 6,041 | 30,159 | 4,219 / 595 / 1,227 | 0 direct binary masks | 6,041 |
| S2Looking | 10,000 | 10,000 | 7,000 / 1,000 / 2,000 | 10,000 | 0 (localization-only) |
| **Total** | **26,113** | **90,519** | — | — | — |

Parser-derived coverage is highly uneven.  LEVIR contains 8,078 captions with
`two`, but only 63 with `five`; SECOND-CC contains 2,585 with `two`, but only
23 with `five`. S2Looking contributes 10,000 template captions for building
appearance, but no location or count phrases and no retrieval supervision.
This makes a claim of learned count understanding premature until verified
coverage is measured.

The initial joint parser table has 11,626 cells; 7,348 are singletons and
10,152 have fewer than five caption rows.  Therefore exact joint inverse
frequency weighting would be unsafe: it would make noisy singleton cells
dominant. Smoothed, capped weights and minimum-coverage eligibility are
required.

Exact `(T1,T2,mask)` path triples did not duplicate across these manifests.
However, 2,427 normalized-caption clusters span more than one pair and 1,117
of those clusters span source split names. This is not proof of visual leakage
(repeated generic captions are expected), but it is a mandatory review item:
future near-duplicate visual clustering and pair-level split checks must decide
whether any cluster is a true leakage risk.
