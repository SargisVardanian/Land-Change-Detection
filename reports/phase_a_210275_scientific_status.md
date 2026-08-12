# Phase A job 210275

Job `210275` completed the runtime contract (`456/456`, exit 0, checkpoint
roundtrip PASS, no NaN/OOM), but it is not a valid scientific calibration
point. The fixed two-caption slot formulation duplicated singleton captions,
and the deterministic singleton audit rejected that weighting contract.

Therefore its permanent status is:

`ENGINEERING_VALID_SCIENTIFICALLY_INVALID_FOR_BUDGET_SELECTION`

The checkpoint is preserved as engineering evidence only and must not
initialize corrected Phase A or Phase B. Corrected Phase A must start from the
same pinned pretrained SigLIP2 NaFlex initialization. Jobs `210272` and
`210274` remain preserved as engineering-failure history.
