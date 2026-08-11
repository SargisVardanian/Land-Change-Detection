# Small post-Phase-B representation audit

Status: `PARTIAL_MISSING_POSTB_EMBEDDING_ARRAYS`

The B20/B24 development ranking tensors are valid for retrieval and route
audits, but the saved run does not contain pair-embedding and text-embedding
arrays. Effective rank, participation ratio, top-5 explained variance, and
source-probe accuracy/AUC are therefore not computed for B20 or B24. A score
matrix is not a substitute for the embedding matrix, so these fields remain
`null` rather than being inferred.

## Contract

- Dataset release SHA: `6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`
- Development queries: 3,181
- Development gallery: 995 physical pairs
- Manifest SHA: `e08e2da0c93a649fcce17d200bf413a7f92443f65fc0901e5ead8961c254a98c`
- Ordered gallery SHA: `1627b745bed8d2e78cc9b99ff1be9f4f792025d8e8a3f275ad0cda199ba554ef`
- Ordered query SHA: `ed25e0f52a93d4e8fa785da3f504a669d79a99962a657550bae1019445d075ff`

## Available representation reference

The Phase-A reference artifact reports pair effective rank `36.90` at step 0
and `8.60` at step 456, with participation ratio `10.16` and `4.49`,
respectively. Pair source-probe accuracy rises from `0.981` to `0.997`.
The text reference remains effective rank `124.54`, participation ratio
`46.06`, top-5 variance `0.264`, and source-probe accuracy `0.876`.

These values are retained as a Phase-A reference, not presented as a fresh
B20/B24 matrix audit.

## Route audit

`ROUTE_GAP_OBSERVED` on the fixed 64-pair/172-query subset:

| checkpoint | route | MRR | Hit@10 |
|---|---|---:|---:|
| B20 | DIRECT_NAFLEX | 0.2595 | 0.5930 |
| B20 | HIERARCHICAL_NATIVE | 0.2331 | 0.5814 |
| B24 | DIRECT_NAFLEX | 0.2603 | 0.5872 |
| B24 | HIERARCHICAL_NATIVE | 0.2350 | 0.5814 |

Same-pair embedding cosine mean is `0.9248` for B20 and `0.9170` for B24.
The route gap is a diagnostic discontinuity only; it is not evidence by
itself of an architecture defect or a reason to redesign the model.

## Limitation

The next run should save the ordered pair and text embedding arrays (or a
documented compact representation sufficient to reconstruct them) if the
post-B effective-rank/source-probe audit is required. No new training or
architecture change is authorized by this report.
