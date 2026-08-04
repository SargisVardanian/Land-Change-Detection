from land_change_detection.training.hierarchical_source_sampler import (
    HierarchicalSamplingConfig,
    HierarchicalSourceSampler,
)


def _rows():
    return [
        {"canonical_pair_id": f"pair-{event}-{i}", "source_dataset": "RSCC-EBD", "source_event_id": event, "change_type": "damage"}
        for event in ("e0", "e1", "e2", "e3")
        for i in range(32)
    ]


def test_event_cap_is_applied_to_every_batch():
    sampler = HierarchicalSourceSampler(_rows(), HierarchicalSamplingConfig(batch_size=16, max_event_fraction=0.25, max_source_fraction=1.0))
    for batch in sampler.sample_epoch(8):
        assert max(sampler.counts(batch).values()) <= 4
        assert len({r["canonical_pair_id"] for r in batch}) == 16


def test_epoch_is_not_event_dominated():
    sampler = HierarchicalSourceSampler(_rows(), HierarchicalSamplingConfig(batch_size=16, max_event_fraction=0.25, max_source_fraction=1.0))
    batches = sampler.sample_epoch(8)
    counts = sampler.counts(row for batch in batches for row in batch)
    assert max(counts.values()) - min(counts.values()) <= 4


def test_schedule_hash_is_deterministic():
    cfg = HierarchicalSamplingConfig(batch_size=16, max_event_fraction=0.25, max_source_fraction=1.0)
    left = HierarchicalSourceSampler(_rows(), cfg).sample_epoch(4)
    right = HierarchicalSourceSampler(_rows(), cfg).sample_epoch(4)
    assert HierarchicalSourceSampler.schedule_sha256(left) == HierarchicalSourceSampler.schedule_sha256(right)



def test_event_identity_does_not_create_disaster_domain():
    rows = [
        {
            "canonical_pair_id": "forest-0",
            "source_dataset": "Forest-Change",
            "source_event_id": "scene-0",
            "change_type": "vegetation_loss",
        }
    ]
    sampler = HierarchicalSourceSampler(
        rows,
        HierarchicalSamplingConfig(batch_size=1, max_event_fraction=1.0, max_source_fraction=1.0),
    )
    assert sampler.rows[0]["_event"] == "scene-0"
    assert sampler.rows[0]["_domain"] == "non_disaster"


def test_explicit_domain_overrides_source_policy():
    rows = [
        {
            "canonical_pair_id": "reviewed-0",
            "source_dataset": "Forest-Change",
            "source_event_id": "scene-0",
            "domain": "disaster",
            "change_type": "fire_damage",
        }
    ]
    sampler = HierarchicalSourceSampler(
        rows,
        HierarchicalSamplingConfig(batch_size=1, max_event_fraction=1.0, max_source_fraction=1.0),
    )
    assert sampler.rows[0]["_domain"] == "disaster"
