# QCPR v3 forensic correction protocol

This correction keeps three independent evidence families. Exact-pair ranking is an instance-identity diagnostic. Broad teacher-semantic relevance is a coarse semantic retrieval diagnostic and is not human ground truth. Structured near misses are parser-derived audit/mining labels unless a separately reviewed benchmark supplies human verification; parser-derived results cannot pass the B-stage scientific gate.

The immutable E0 checkpoint is evaluated only through `scripts/audit_qcpr_e0_full.py` against the complete natural gallery. The report separates duplicate-aware, teacher-semantic, changed/no-change, frequent/rare caption clusters and non-visual text/no-change baselines.

For Stage B, the v1 global path remains frozen. The late-interaction score aggregates top-k temporal patch evidence per valid content token and averages across content tokens; special tokens and stopwords are excluded. Hard negatives are selected only from global top-N, after broad semantic positives have been excluded. Parser categories are used to mine wrong-object, wrong-direction, wrong-location, wrong-count and change/no-change conflicts, but they are not model heads and do not make the gate human verified.

Stage C remains disabled. Its contract uses one query-conditioned logits field for rendered masks, masked pooling, reranking and losses. Targets are resized with foreground-preserving pooling and are recorded separately for generic change, query-specific, appeared and disappeared supervision, including mismatched-query empty-mask negatives.

The B sampler is pair-level and deterministic: caption paraphrases are selected inside the collator, no-change rows are capped per batch, duplicate-caption mass is downweighted, and all sample weights remain capped. Natural validation is never rebalanced.
