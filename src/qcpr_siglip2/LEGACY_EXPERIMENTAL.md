# Legacy experimental SigLIP2 path

The modules in this package reproduce the bounded historical Phase-A runs.
They include the former evidence bottleneck and Top-K reranking experiments.
They are preserved for checkpoint and ranking reproducibility only.

The active direct retrieval implementation is now:

```text
src/qcpr_temporal_siglip/
```

New TemporalSigLIP training and inference must not import
`qcpr_siglip2.models.model.Siglip2TemporalRetrievalModel`.
