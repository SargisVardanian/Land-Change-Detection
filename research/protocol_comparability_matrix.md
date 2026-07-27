# QCPR protocol comparability

| Protocol | Temporal pair | Exact physical ID | Semantic multi-positive | Spatial output | Comparable to FULL_GALLERY_EXACT |
|---|---:|---:|---:|---:|---:|
| FULL_GALLERY_EXACT | yes | yes | optional ignored set | no | reference |
| SEMANTIC_MULTI_POSITIVE | yes | no | yes | no | secondary only |
| CHANGERETCAP_COMPAT | yes | no | paper-defined | no | no |
| TEXT_ITSR_SEMANTIC_COMPAT | yes | no | caption similarity | no | no |
| AIR_SLT_COMPAT | no | no | region points | yes | no |

Headline values from papers must remain in their native protocol. A larger
number in a different protocol is not evidence of a better QCPR retriever.
