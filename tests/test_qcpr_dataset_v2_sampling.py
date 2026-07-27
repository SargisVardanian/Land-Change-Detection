import pytest
from land_change_detection.training.qcpr_dataset_v2_sampling import MixturePolicy,sample_weight,validate_batch

def row(pair, **extra):
    return {"canonical_pair_id":pair,"query_scope":"exact_pair","is_generated":False,"is_synthetic":False,**extra}

def test_generic_no_change_is_downweighted():
    assert sample_weight(row("a",query_scope="generic_no_change")) < sample_weight(row("b"))

def test_unverified_generated_is_lower_trust():
    assert sample_weight(row("a",is_generated=True,verification_status="unverified")) < sample_weight(row("b"))

def test_batch_rejects_duplicate_pair_and_synthetic_overflow():
    with pytest.raises(ValueError): validate_batch([row("a"),row("a")])
    with pytest.raises(ValueError): validate_batch([row(str(i),is_synthetic=True) for i in range(4)],MixturePolicy(synthetic_max_fraction=.3))
