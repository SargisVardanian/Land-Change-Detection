from qcpr_data.queries.purpose import QUERY_PURPOSES, classify_query, semantic_group_id


def test_classifier_returns_exactly_one_supported_purpose():
    cases = [
        ({"query_scope": "exact_pair", "verification_status": "human", "text": "a new road appears on the left"}, "exact_discriminative"),
        ({"query_scope": "generic_no_change", "change_status": "no_change", "text": "the two images are the same"}, "generic_no_change"),
        ({"query_scope": "direction", "verification": "human", "text": "buildings appeared"}, "direction_sensitive"),
        ({"query_scope": "localized", "verification": "derived_eval", "text": "new buildings near the bottom"}, "localized"),
        ({"query_scope": "long_series", "verification": "generated_unverified", "text": "development grows over time"}, "long_series"),
    ]
    for row, expected in cases:
        result = classify_query(row, positive_set_size=1, collision_count=1, neighbour_item_count=1)
        assert result["purpose"] in QUERY_PURPOSES
        assert result["purpose"] == expected


def test_generic_no_change_is_not_stable():
    result = classify_query({"query_scope": "generic_no_change", "change_status": "no_change", "text": "there is no difference"})
    assert result["purpose"] == "generic_no_change"


def test_stable_scene_requires_common_anchors_and_is_pair_discriminative():
    row = {"query_scope": "stable_scene_candidate", "change_status": "no_change", "verification": "visual_probe_candidate", "text": "The same road and buildings remain visible in both observations."}
    result = classify_query(row, positive_set_size=1, stable_anchor_count=2, stable_identifiability=0.8)
    assert result["purpose"] == "stable_scene_specific"
    non_unique = classify_query(row, positive_set_size=2, stable_anchor_count=2, stable_identifiability=0.8)
    assert non_unique["purpose"] == "semantic_multi_positive"


def test_semantic_group_id_excludes_event_and_source_identity():
    left = {"objects": ["building"], "directions": ["appearance"], "change_types": ["appearance"], "spatial_relations": [], "surfaces": ["built"]}
    right = dict(left)
    assert semantic_group_id(left) == semantic_group_id(right)


def test_causal_or_severity_claim_is_rejected_when_unverified():
    result = classify_query({"query_scope": "exact_pair", "verification": "source_unverified", "text": "severe loss occurred because of a flood"})
    assert result["purpose"] == "unsupported_or_reject"
