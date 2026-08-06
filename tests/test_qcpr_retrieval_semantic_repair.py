import json

from build_qcpr_retrieval_semantic_repair import build_official_source_acquisition_audit
from package_qcpr_retrieval_semantic_repair import write_relevance_graph
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


def test_no_alteration_is_generic_when_unattributed():
    result = classify_query({"query_scope": "exact_pair", "verification": "human", "text": "there is no alteration"})
    assert result["purpose"] == "generic_no_change"


def test_exact_requires_supported_visual_attribute():
    result = classify_query(
        {
            "query_scope": "exact_pair",
            "verification": "human",
            "text": "white rectangular shapes are present in the open area",
        },
        positive_set_size=1,
        collision_count=1,
        neighbour_item_count=1,
    )
    assert result["purpose"] == "unsupported_or_reject"


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


def test_relevance_graph_materializes_all_positive_edges(tmp_path):
    rows = [
        {
            "query_id": "q2",
            "query_scope": "generic_no_change",
            "purpose": "generic_no_change",
            "source_item_id": "item-b",
            "positive_item_ids": ["item-b"],
            "graded_relevance": {"item-b": 3},
            "split": "development",
            "verification": "human",
            "training_enabled": False,
        },
        {
            "query_id": "q1",
            "query_scope": "semantic",
            "purpose": "semantic_multi_positive",
            "source_item_id": "item-a",
            "positive_item_ids": ["item-c", "item-a"],
            "graded_relevance": {"item-a": 3, "item-c": 2},
            "split": "train",
            "verification": "human",
            "training_enabled": False,
            "provenance": {"semantic_group_id": "group-1"},
        },
    ]
    path = tmp_path / "registries/relevance_graph.jsonl"
    metadata = write_relevance_graph(path, rows)
    edges = [json.loads(line) for line in path.read_text().splitlines()]
    assert metadata["edge_count"] == 3
    assert [(edge["query_id"], edge["item_id"]) for edge in edges] == [
        ("q1", "item-a"),
        ("q1", "item-c"),
        ("q2", "item-b"),
    ]
    assert [edge["relevance_grade"] for edge in edges] == [3, 2, 3]
    assert edges[-1]["diagnostic_only"] is True
    assert edges[0]["semantic_group_id"] == "group-1"


def test_source_acquisition_audit_does_not_promote_raw_or_adapter_stubs(tmp_path):
    project = tmp_path / "project"
    release = project / "manifests" / "release"
    release.mkdir(parents=True)
    raw = project / "datasets" / "raw" / "RSRCC"
    raw.mkdir(parents=True)
    (raw / "before.png").write_bytes(b"raw")
    code = tmp_path / "code"
    adapter = code / "src" / "qcpr_data" / "sources"
    adapter.mkdir(parents=True)
    (adapter / "rsrcc.py").write_text(
        'ADAPTER = RegistrySourceAdapter("RSRCC", "unacquired")\n',
        encoding="utf-8",
    )
    (release / "source_reports").mkdir()
    (release / "source_reports" / "rsrcc_source_audit.json").write_text("{}", encoding="utf-8")
    audit = build_official_source_acquisition_audit(
        current_release=release,
        code_repo=code,
        prior={
            "blocked_or_deferred": [
                {
                    "source_dataset": "RSRCC",
                    "license_status": "APACHE_2.0_SOURCE_TERMS_AND_PARENT_DATA_REVIEW_REQUIRED",
                }
            ]
        },
    )
    row = audit["sources"]["RSRCC"]
    assert row["physical_assets"] is True
    assert row["loader_available"] is True
    assert row["loader"] is False
    assert row["hashes"] is False
    assert row["license"] is False
    assert row["manifest"] is False
    assert row["ready_for_integration"] is False
    assert audit["ready_sources"] == []
