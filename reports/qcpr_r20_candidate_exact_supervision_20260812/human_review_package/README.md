# QCPR r20 exact-supervision human review package

Status: **READY_FOR_TWO_INDEPENDENT_HUMAN_REVIEWERS**

This package is derived from immutable r19g release `6d90faeded7a0cadfd9e0a46b48c9d6b9d623cea2d9cbd7439b07bedc62ab871`.
It contains 1,500 rows: 300 exact-discriminative (150 LEVIR + 150 SECOND, with
unique physical pairs), 300 semantic-multi-positive,
300 generic-no-change, 300 stable-scene-specific, and 300 localized-direction.

Review T1 and T2 for the intended pair, then inspect the candidate neighbours.
The sheets contain no model scores, rankings, masks, event IDs, or dataset/source
names. Candidate neighbours are sampling aids only and are never automatically
positive or negative. Do not infer facts from paths or IDs.

For each row answer independently:

1. Does the caption accurately describe the intended T1→T2 pair?
2. Does it identify this pair rather than many alternatives?
3. Is any candidate neighbour also fully consistent with the caption?
4. Choose exactly one: `EXACT, SEMANTIC_ONLY, AMBIGUOUS_IGNORE, REWRITE_REQUIRED, REJECT`.

`REWRITE_REQUIRED` captions must be rewritten using visible T1/T2 evidence only.
No masks, coordinates unavailable to a human viewer, event metadata, source names,
or model output may be used. Two distinct human identities, timestamps, and
independence attestations are required. Codex/agent inspection is not human review.

Files:

- `reviewer_a_packet.jsonl`, `reviewer_b_packet.jsonl`: independent visual sheets
- `reviewer_a_decision_template.jsonl`, `reviewer_b_decision_template.jsonl`: decision sheets
- `adjudication_template.jsonl`: final adjudication after both sheets are returned
- `review_package_audit.json`: lineage and hash audit

No r20 training manifest is created by this package. Promotion remains disabled
until human decisions and adjudication pass the contract.
