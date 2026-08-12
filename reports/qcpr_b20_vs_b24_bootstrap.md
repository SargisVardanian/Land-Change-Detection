# B20 vs B24 and Top-20 Audit

Status: **PASS_MANIFEST_MATCH_STALE_RUN_RELEASE_FIELD**

The paired bootstrap unit is the physical pair; all query rows for a sampled pair are retained together.
The per-query export contains both B20 (step 1140) and B24 (step 1368) on the same r19g gallery.

## Inputs

- authoritative release SHA: `6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`
- development manifest SHA: `e08e2da0c93a649fcce17d200bf413a7f92443f65fc0901e5ead8961c254a98c`
- gallery IDs SHA: `1627b745bed8d2e78cc9b99ff1be9f4f792025d8e8a3f275ad0cda199ba554ef`
- query IDs SHA: `ed25e0f52a93d4e8fa785da3f504a669d79a99962a657550bae1019445d075ff`
- gallery/query: `995` / `3181`
- historical run release field match: `False`

## Point metrics

| Arm | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@50 | Hit@100 | mean rank | median rank | within MRR | within Hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B20 | 0.07503565 | 0.02232002 | 0.09965420 | 0.16567117 | 0.46840617 | 0.65608299 | 104.1701 | 57.0 | 0.07790561 | 0.17353034 |
| B24 | 0.07586470 | 0.02420622 | 0.09871110 | 0.16567117 | 0.46526250 | 0.65702611 | 105.7557 | 58.0 | 0.07836429 | 0.17038667 |

## Bootstrap

- replicates: `5000`; seed: `20260812`; delta: `B24 - B20`.
- See the JSON for percentile 95% intervals and empirical two-sided sign p-values.

## Integrity

- B20 Top-100 cross-check: `{'query_rows': 3181, 'serialized_rows': 3181, 'missing_queries': 0, 'malformed_rows': 0, 'status': 'PASS', 'examples': {'missing': [], 'malformed': []}}`
- B24 Top-100 cross-check: `{'query_rows': 3181, 'serialized_rows': 3181, 'missing_queries': 0, 'malformed_rows': 0, 'status': 'PASS', 'examples': {'missing': [], 'malformed': []}}`
- Semantic multi-positive evaluation is not performed because the current dataset handoff reports semantic evaluation not ready.
- Historical run metadata records a stale release-root hash, but the exact development manifest, ordered query IDs and ordered gallery IDs match the authoritative r19g inputs. This is retained as a provenance warning, not silently hidden.
