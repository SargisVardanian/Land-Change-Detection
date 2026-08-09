# QCPR Dataset-v2 data dictionary

## Physical item

| Field | Meaning |
|---|---|
| `item_id` | Stable physical pair/sequence identity. |
| `item_type` | `pair` or `sequence`. |
| `source`, `source_revision` | Dataset lineage, never a relevance label. |
| `physical_group_id` | Leakage boundary for related physical content. |
| `scene_id`, `event_id` | Scene/event metadata; event may be null. |
| `frames` | Ordered native frame records. |
| `split` | `train`, `development`, or `test`. |
| `training_enabled` | Physical eligibility, not sufficient text authorization. |
| `quality_status` | Physical QA state. |
| `provenance` | Source archive, parsing and lineage evidence. |

## Frame

| Field | Meaning |
|---|---|
| `frame_id` | Stable frame identity. |
| `path` | Absolute native asset path on the cluster. |
| `sha256` | Byte identity of the native asset. |
| `timestamp` | Ordered time label; may be source-relative. |
| `sensor` | Nullable sensor metadata. |
| `gsd` | Nullable ground sample distance; unknown is null. |
| `width`, `height` | Native decoded dimensions. |

## Canonical query

| Field | Meaning |
|---|---|
| `query_id` | Stable language identity. |
| `text` | Factual retrieval text. |
| `query_scope` | Task projection: exact, semantic, localized, direction, stable or long_series. |
| `query_classification` | Exactly one retrieval-purpose class. |
| `source_item_id`, `source_pair_id` | Physical origin of the text. |
| `positive_item_ids` | Verified positive physical items. |
| `graded_relevance` | Sparse item-to-grade mapping (0-3). |
| `temporal_direction` | Direction field; `none` when unsupported. |
| `localized_relation` | Verified region relation or null. |
| `verification` | Human/generated/rule/eval provenance state. |
| `training_enabled` | Final projection-local training gate. |
| `candidate_training_enabled` | Candidate eligibility only; never overrides final gate. |
| `training_gate` | Evidence gate or hold reason. |
| `split` | Query projection split. |
| `roles` | Non-exclusive capabilities such as directional or localized. |
| `attributes` | Objects, direction, change type, surfaces and spatial relations. |
| `caption_provenance`, `provenance` | Text origin and audit lineage. |

## Relevance and collisions

`registries/relevance_graph.jsonl` is sparse and non-transitive. Grade 3 is strongly
relevant; grades 1-2 are semantic/ambiguous and are ignored in exact loss unless
independently verified; grade 0 is an explicit negative. `collision_groups.jsonl`
groups normalized duplicate or ambiguous queries so loaders can construct ignore
cells without a dense global matrix.

## Views and sidecars

`manifests/<scope>_<split>.jsonl` are immutable projections over canonical identities.
Projection metadata does not redefine canonical text. Dense masks/labels are stored in
`evaluation_sidecars/dense.jsonl` under the mask-sidecar contract and always have
`evaluation_only=true`; they must never generate retrieval text or exact-loss targets.

