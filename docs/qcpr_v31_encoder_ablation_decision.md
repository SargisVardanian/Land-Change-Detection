# QCPR v3.1 encoder ablation decision

Run:
`runs/qcpr_v31_encoder_ablation_369b0c02_siglipfix_20260718-000209`

Slurm job `104760` completed `0:0` on the fixed natural validation contract:
1,928 pairs, 9,640 queries, manifest SHA256
`6fec17209c4d02fc0cba81b322cec80990f7363e9a6ce3e8a945fe2834c28ae1`.

The semantic relevance labels are text-derived pseudo targets, not
human-verified image relevance. Exact-pair metrics remain diagnostic.

| Encoder/score contract | R@1 | R@5 | R@10 | nDCG@10 | recall@50 | recall@100 | recall@200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned Jina+UniverSat pair anchor | .5438 | .7974 | .8974 | .4476 | .9380 | .9485 | .9973 |
| GeoRSCLIP zero-shot signed delta | .5422 | .8591 | .9224 | .5271 | .9658 | .9803 | .9965 |
| RemoteCLIP zero-shot signed delta | .3945 | .8537 | .9199 | .4832 | .9714 | .9860 | .9964 |
| SigLIP2 zero-shot signed delta | .3622 | .8534 | .9145 | .5158 | .9539 | .9672 | .9884 |

The zero-shot alternatives do not have the learned bi-temporal pair head and
are therefore not architecture-compatible global replacements. The production
candidate generator remains the learned Jina+UniverSat path.

GeoRSCLIP is the strongest zero-shot temporal retrieval alternative, but its
ViT-B/32 dense field is only 7x7. RemoteCLIP has the same localization
resolution. SigLIP2 provides a 16x16 dense field, aligned text tokens, and a
pretraining design intended to improve dense prediction. It is selected only
as a frozen grounding feature path.

The next experiment keeps global retrieval unchanged and replaces only the
grounding image/text tokens with frozen SigLIP2 features. All trainable
components remain generic temporal patch projection, cross-modal interaction,
and the query-conditioned FPN. No semantic-specific head is introduced.
